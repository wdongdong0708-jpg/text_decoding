from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_keyword_decoding.data import (
    ContextEEGCollator,
    ContextEEGDataset,
    build_lexical_identity_index,
    build_master_keyword_index,
    build_split_view_index,
    build_subject_index,
)
from eeg_keyword_decoding.evaluation import (
    TextFreePrototypeScorer,
    load_fold_keyword_eligibility,
)
from eeg_keyword_decoding.losses import ContextOTThreeLoss, ThreeLossConfig
from eeg_keyword_decoding.models import (
    build_context_text_projector,
    build_eeg_sequence_encoder,
)
from eeg_keyword_decoding.ot import (
    ContextOTAligner,
    MaskedBalancedSinkhorn,
    MaskedBalancedSinkhornConfig,
    TransportPoolingConfig,
)
from eeg_keyword_decoding.prototypes import (
    PrototypeBuilderConfig,
    TrainOnlyPrototypeBuilder,
    file_sha256,
    prototype_bank_sha256,
)
from eeg_keyword_decoding.training import (
    EarlyStopping,
    FoldTrainer,
    FoldTrainerConfig,
    OptimizerConfig,
    PrototypeRefreshCoordinator,
    ReproducibilityConfig,
    SchedulerConfig,
    TrainEpochConfig,
    build_adamw_optimizer,
    build_scheduler,
    configure_reproducibility,
    create_unique_run_directory,
    make_dataloader_generator,
    seed_dataloader_worker,
)


PROTOCOL = PROJECT_ROOT / "data/protocols/littleprince_hf_v1"
MANIFEST = PROJECT_ROOT / "data/manifests/littleprince_pl_all_clean_manifest.csv"
FOLDS = PROTOCOL / "littleprince_sentence_folds_v1.csv"
ELIGIBILITY = PROTOCOL / "littleprince_keyword_fold_eligibility_v1.csv"
LEXICON = PROTOCOL / "littleprince_hf_lexicon_v1.csv"
LABELS = PROTOCOL / "littleprince_sentence_keyword_labels_v1.csv"
OCCURRENCES = PROTOCOL / "littleprince_word_occurrences_v1.csv"
CACHE_DIRS = {
    "macbert": PROJECT_ROOT / "data/cache/context_words/macbert_v1",
    "bge_m3": PROJECT_ROOT / "data/cache/context_words/bge_m3_colbert_v1",
}
TEXT_CONFIGS = {
    "macbert": PROJECT_ROOT / "configs/text/macbert_projection_v1.yaml",
    "bge_m3": PROJECT_ROOT / "configs/text/bge_m3_projection_v1.yaml",
}


def _device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return result


def _load_config(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != "fold_training_v1":
        raise ValueError("Unsupported training config")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leakage-guarded, non-scientific single-fold training smoke."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/training/fold_train_smoke_v1.yaml",
    )
    parser.add_argument("--outer-fold", type=int, default=None)
    parser.add_argument(
        "--text-backend", choices=("macbert", "bge_m3"), default=None
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-validation-batches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--precision", choices=("fp32", "amp"), default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    config = _load_config(args.config.resolve())
    outer_fold = (
        int(config.get("outer_fold", 0))
        if args.outer_fold is None
        else args.outer_fold
    )
    backend = config["text_backend"] if args.text_backend is None else args.text_backend
    epochs = int(config["epochs"] if args.epochs is None else args.epochs)
    batch_size = int(
        config["batch_size"] if args.batch_size is None else args.batch_size
    )
    num_workers = int(
        config["num_workers"] if args.num_workers is None else args.num_workers
    )
    precision = config["precision"] if args.precision is None else args.precision
    max_train_batches = (
        config.get("max_train_batches")
        if args.max_train_batches is None
        else args.max_train_batches
    )
    max_validation_batches = (
        config.get("max_validation_batches")
        if args.max_validation_batches is None
        else args.max_validation_batches
    )
    device = _device(args.device)
    if precision == "amp" and device.type != "cuda":
        raise ValueError("AMP smoke requires CUDA")
    if outer_fold not in range(5):
        raise ValueError("outer-fold must be in [0,4]")
    if not config.get("smoke_run") or not config.get(
        "not_for_scientific_reporting"
    ):
        raise ValueError("This command only runs explicitly marked smoke configs")

    reproducibility = ReproducibilityConfig(
        seed=int(config["reproducibility"]["seed"]),
        deterministic_algorithms=bool(
            config["reproducibility"]["deterministic_algorithms"]
        ),
        cudnn_benchmark=bool(config["reproducibility"]["cudnn_benchmark"]),
    )
    configure_reproducibility(reproducibility)
    generator = make_dataloader_generator(reproducibility.seed)

    split_index = build_split_view_index(MANIFEST, FOLDS)
    subject_index = build_subject_index(split_index.manifest_records)
    keyword_index = build_master_keyword_index(LEXICON)
    lexical_index = build_lexical_identity_index(OCCURRENCES, split_index)
    common = {
        "split_index": split_index,
        "outer_fold": outer_fold,
        "subject_index": subject_index,
        "keyword_index": keyword_index,
        "lexical_identity_index": lexical_index,
        "sentence_labels_path": LABELS,
        "word_occurrences_path": OCCURRENCES,
        "expected_eeg_channels": 128,
    }
    # Deliberately instantiate exactly train and validation. There is no test branch.
    train_dataset = ContextEEGDataset(
        **common,
        role="train",
        text_backend=backend,
        context_store_root=CACHE_DIRS[backend],
        include_context_targets=True,
    )
    validation_dataset = ContextEEGDataset(
        **common,
        role="validation",
        include_context_targets=False,
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": ContextEEGCollator(),
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_dataloader_worker,
        "persistent_workers": num_workers > 0,
        "multiprocessing_context": "spawn" if num_workers > 0 else None,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    eeg_encoder = build_eeg_sequence_encoder(
        PROJECT_ROOT / "configs/models/eeg_sequence_conv_v1.yaml",
        actual_input_channels=128,
        actual_num_subjects=len(subject_index.subjects),
    )
    text_projector = build_context_text_projector(
        TEXT_CONFIGS[backend],
        cache_metadata_path=CACHE_DIRS[backend] / "metadata.json",
    )
    sinkhorn_config = MaskedBalancedSinkhornConfig.from_yaml(
        PROJECT_ROOT / "configs/ot/masked_balanced_sinkhorn_v1.yaml"
    )
    transport_config = TransportPoolingConfig.from_yaml(
        PROJECT_ROOT / "configs/ot/transport_pooling_v1.yaml"
    )
    aligner = ContextOTAligner(
        text_projector=text_projector,
        sinkhorn=MaskedBalancedSinkhorn.from_config(sinkhorn_config),
        transport_eps=transport_config.eps,
    )
    loss_config = ThreeLossConfig.from_yaml(
        PROJECT_ROOT / "configs/losses/context_ot_three_loss_v1.yaml"
    )
    loss_module = ContextOTThreeLoss(loss_config)
    prototype_builder = TrainOnlyPrototypeBuilder(
        config=PrototypeBuilderConfig.from_yaml(
            PROJECT_ROOT / "configs/prototypes/train_only_group_balanced_v1.yaml"
        ),
        split_index=split_index,
        outer_fold=outer_fold,
        keyword_index=keyword_index,
        lexical_identity_index=lexical_index,
        sentence_labels_path=LABELS,
        word_occurrences_path=OCCURRENCES,
        eligibility_path=ELIGIBILITY,
        context_store_root=CACHE_DIRS[backend],
        text_backend=backend,
    )
    eligibility = load_fold_keyword_eligibility(
        ELIGIBILITY, outer_fold=outer_fold, keyword_index=keyword_index
    )
    optimizer_config = OptimizerConfig(
        name=config["optimizer"]["name"],
        eeg_lr=float(config["optimizer"]["eeg_lr"]),
        text_projection_lr=float(config["optimizer"]["text_projection_lr"]),
        scalar_mix_lr=float(config["optimizer"]["scalar_mix_lr"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    optimizer = build_adamw_optimizer(
        eeg_encoder=eeg_encoder,
        text_projector=text_projector,
        config=optimizer_config,
    )
    selected_batches = min(
        len(train_loader),
        int(max_train_batches) if max_train_batches is not None else len(train_loader),
    )
    accumulation = int(config["gradient_accumulation_steps"])
    total_steps = epochs * math.ceil(selected_batches / accumulation)
    warmup_steps = min(
        total_steps - 1,
        int(round(total_steps * float(config["scheduler"]["warmup_ratio"]))),
    )
    scheduler = build_scheduler(
        optimizer,
        SchedulerConfig(
            name=config["scheduler"]["name"],
            total_optimizer_steps=total_steps,
            warmup_steps=warmup_steps,
        ),
    )
    scaler = torch.amp.GradScaler(
        device.type,
        init_scale=float(config.get("amp_initial_scale", 1024.0)),
        enabled=precision == "amp" and device.type == "cuda",
    )

    resolved = dict(config)
    resolved.update(
        {
            "outer_fold": outer_fold,
            "text_backend": backend,
            "epochs": epochs,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "precision": precision,
            "max_train_batches": max_train_batches,
            "max_validation_batches": max_validation_batches,
            "total_optimizer_steps": total_steps,
            "warmup_steps": warmup_steps,
            "train_dataloader_length": len(train_loader),
            "validation_dataloader_length": len(validation_loader),
            "eeg_model": eeg_encoder.to_config(),
            "text_projection": text_projector.config.canonical_sha256,
            "loss": loss_config.to_metadata(),
            "sinkhorn": sinkhorn_config.__dict__,
            "transport": transport_config.__dict__,
        }
    )
    assets = {
        "fold_file": file_sha256(FOLDS),
        "eligibility_file": file_sha256(ELIGIBILITY),
        "word_occurrences": file_sha256(OCCURRENCES),
        "sentence_keyword_labels": file_sha256(LABELS),
        "eeg_manifest": file_sha256(MANIFEST),
        "lexicon": file_sha256(LEXICON),
        "context_cache_metadata": file_sha256(
            CACHE_DIRS[backend] / "metadata.json"
        ),
    }
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else (PROJECT_ROOT / config["output_root"]).resolve()
    )
    run_directory = (
        args.resume.resolve().parent.parent
        if args.resume is not None
        else create_unique_run_directory(
            output_root, backend=backend, outer_fold=outer_fold
        )
    )
    run_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )
    provenance = {
        "asset_hashes": assets,
        "subject_index_mapping": subject_index.subject_to_index,
        "keyword_index_mapping": keyword_index.keyword_to_index,
        "train_sentence_count": len(
            split_index.sentence_indices_for(outer_fold, "train")
        ),
        "train_eeg_view_count": len(train_dataset),
        "validation_sentence_count": len(
            split_index.sentence_indices_for(outer_fold, "validation")
        ),
        "validation_eeg_view_count": len(validation_dataset),
        "test_dataset_created": False,
        "smoke_run": True,
        "not_for_scientific_reporting": True,
    }
    (run_directory / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    trainer = FoldTrainer(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        train_loader=train_loader,
        validation_loader=validation_loader,
        eeg_encoder=eeg_encoder,
        text_projector=text_projector,
        aligner=aligner,
        loss_module=loss_module,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        prototype_refresh=PrototypeRefreshCoordinator(prototype_builder),
        validation_scorer=TextFreePrototypeScorer(
            temperature=float(config["validation_scorer"]["temperature"])
        ),
        eligibility=eligibility,
        early_stopping=EarlyStopping(
            patience=int(config["early_stopping"]["patience"]),
            min_delta=float(config["early_stopping"]["min_delta"]),
            metric=config["early_stopping"]["metric"],
            mode=config["early_stopping"]["mode"],
        ),
        config=FoldTrainerConfig(
            epochs=epochs,
            train_epoch=TrainEpochConfig(
                precision=precision,
                gradient_clip_norm=float(config["gradient_clip_norm"]),
                gradient_accumulation_steps=accumulation,
                max_batches=(
                    int(max_train_batches)
                    if max_train_batches is not None
                    else None
                ),
            ),
            max_validation_batches=(
                int(max_validation_batches)
                if max_validation_batches is not None
                else None
            ),
            smoke_run=True,
            not_for_scientific_reporting=True,
        ),
        resolved_config=resolved,
        asset_hashes=assets,
        subject_index_mapping=subject_index.subject_to_index,
        keyword_index_mapping=keyword_index.keyword_to_index,
        run_directory=run_directory,
        device=device,
        dataloader_generator=generator,
        project_root=PROJECT_ROOT,
    )
    result = trainer.fit(
        resume_from=args.resume,
        stop_after_epoch=args.stop_after_epoch,
    )
    validation = result.final_validation
    summary = {
        **provenance,
        "outer_fold": outer_fold,
        "backend": backend,
        "precision": precision,
        "device": str(device),
        "subject_adapter_enabled": eeg_encoder.config.subject_adapter.enabled,
        "completed_epoch": result.completed_epoch,
        "global_optimizer_step": result.global_optimizer_step,
        "best_epoch": result.best_epoch,
        "best_metric": result.best_metric,
        "prototype_available_count": result.final_prototype_bank.available_count,
        "prototype_bank_hash": prototype_bank_sha256(
            result.final_prototype_bank
        ),
        "projector_hash": result.final_prototype_bank.projector_state_hash,
        "validation_counts": {
            level: table.scores.shape[0]
            for level, table in validation.tables.items()
        },
        "core_macro_auprc": {
            level: validation.reports[f"core/{level}"].macro_auprc
            for level in validation.tables
        },
        "valid_core_keyword_count": validation.reports[
            "core/context_group"
        ].valid_keyword_count,
        "diagnostic_keyword_coverage": {
            tier: {
                "valid": validation.reports[
                    f"{tier}/context_group"
                ].valid_keyword_count,
                "eligible": validation.reports[
                    f"{tier}/context_group"
                ].eligible_keyword_count,
                "tier_total": validation.reports[
                    f"{tier}/context_group"
                ].total_tier_keyword_count,
            }
            for tier in ("core", "main", "extended")
        },
        "train_epochs": [
            {
                **item.__dict__,
                "peak_gpu_memory_mib": item.peak_gpu_memory_bytes / 2**20,
            }
            for item in result.train_epoch_results
        ],
        "last_checkpoint": str(result.last_checkpoint),
        "best_checkpoint": str(result.best_checkpoint),
        "resume_from": str(args.resume) if args.resume is not None else None,
        "test_dataset_created": result.test_dataset_created,
    }
    (run_directory / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
