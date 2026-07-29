from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset


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
from eeg_keyword_decoding.losses import (
    ContextOTThreeLoss,
    ThreeLossConfig,
)
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
)


PROTOCOL = PROJECT_ROOT / "data" / "protocols" / "littleprince_hf_v1"
MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "littleprince_pl_all_clean_manifest.csv"
)
CACHE_DIRS = {
    "macbert": PROJECT_ROOT / "data" / "cache" / "context_words" / "macbert_v1",
    "bge_m3": (
        PROJECT_ROOT
        / "data"
        / "cache"
        / "context_words"
        / "bge_m3_colbert_v1"
    ),
}
TEXT_CONFIGS = {
    "macbert": PROJECT_ROOT / "configs" / "text" / "macbert_projection_v1.yaml",
    "bge_m3": PROJECT_ROOT / "configs" / "text" / "bge_m3_projection_v1.yaml",
}
DEFAULT_EEG_CONFIG = (
    PROJECT_ROOT / "configs" / "models" / "eeg_sequence_conv_v1.yaml"
)
DEFAULT_SINKHORN_CONFIG = (
    PROJECT_ROOT / "configs" / "ot" / "masked_balanced_sinkhorn_v1.yaml"
)
DEFAULT_TRANSPORT_CONFIG = (
    PROJECT_ROOT / "configs" / "ot" / "transport_pooling_v1.yaml"
)
DEFAULT_LOSS_CONFIG = (
    PROJECT_ROOT / "configs" / "losses" / "context_ot_three_loss_v1.yaml"
)
DEFAULT_PROTOTYPE_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "prototypes"
    / "train_only_group_balanced_v1.yaml"
)


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _autocast(device: torch.device, precision: str):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=precision == "amp",
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _scenario_indices(
    dataset: ContextEEGDataset,
    *,
    scenario: str,
    required: int,
) -> list[int]:
    sentence_by_idx = {
        sentence.text_embedding_idx: sentence
        for sentence in dataset.sentences
    }
    if scenario == "standard":
        return list(range(required))
    if scenario == "no_master":
        candidates = [
            index
            for index, record in enumerate(dataset.records)
            if not dataset._present_keyword_indices[record.text_embedding_idx]
        ]
    elif scenario == "repeated_word":
        repeated_sentences = {
            sentence.text_embedding_idx
            for sentence in dataset.sentences
            if max(
                Counter(
                    item.surface_form for item in sentence.occurrences
                ).values()
            )
            > 1
        }
        candidates = [
            index
            for index, record in enumerate(dataset.records)
            if record.text_embedding_idx in repeated_sentences
        ]
    elif scenario == "multi_view":
        by_sentence: dict[int, list[int]] = {}
        for index, record in enumerate(dataset.records):
            by_sentence.setdefault(record.text_embedding_idx, []).append(index)
        candidates = []
        for _, indices in sorted(
            by_sentence.items(),
            key=lambda item: (-len(item[1]), item[0]),
        ):
            candidates.extend(indices)
            if len(candidates) >= required:
                break
    elif scenario == "maximum_length":
        word_counts = {
            index: sentence_by_idx[index].word_count
            for index in sentence_by_idx
        }
        candidates = sorted(
            range(len(dataset.records)),
            key=lambda index: (
                dataset.records[index].n_samples,
                word_counts[dataset.records[index].text_embedding_idx],
            ),
            reverse=True,
        )
    else:
        raise ValueError(f"Unknown scenario: {scenario!r}")
    if len(candidates) < required:
        raise ValueError(
            f"Scenario {scenario!r} provides only {len(candidates)} samples"
        )
    return candidates[:required]


def _gradient_summary(module: torch.nn.Module) -> dict[str, Any]:
    gradients = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    return {
        "parameter_tensors_with_gradient": len(gradients),
        "all_finite": bool(
            gradients
            and all(bool(torch.isfinite(value).all()) for value in gradients)
        ),
        "maximum_absolute_gradient": (
            max(float(value.detach().abs().max()) for value in gradients)
            if gradients
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real inner-train forward/backward smoke for all three losses."
    )
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--role", choices=("train",), default="train")
    parser.add_argument(
        "--text-backend",
        choices=("macbert", "bge_m3"),
        required=True,
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision",
        choices=("fp32", "amp"),
        default="fp32",
    )
    parser.add_argument(
        "--scenario",
        choices=(
            "standard",
            "no_master",
            "repeated_word",
            "multi_view",
            "maximum_length",
        ),
        default="standard",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eeg-config", type=Path, default=DEFAULT_EEG_CONFIG)
    parser.add_argument("--loss-config", type=Path, default=DEFAULT_LOSS_CONFIG)
    parser.add_argument(
        "--prototype-config",
        type=Path,
        default=DEFAULT_PROTOTYPE_CONFIG,
    )
    args = parser.parse_args()
    if args.outer_fold not in range(5):
        raise ValueError("outer-fold must lie in [0,4]")
    if args.batch_size <= 0 or args.num_batches <= 0:
        raise ValueError("batch-size and num-batches must be positive")
    device = _resolve_device(args.device)
    if args.precision == "amp" and device.type != "cuda":
        raise ValueError("AMP smoke requires CUDA")
    torch.manual_seed(args.seed)

    split_index = build_split_view_index(
        MANIFEST,
        PROTOCOL / "littleprince_sentence_folds_v1.csv",
    )
    subject_index = build_subject_index(split_index.manifest_records)
    keyword_index = build_master_keyword_index(
        PROTOCOL / "littleprince_hf_lexicon_v1.csv"
    )
    lexical_index = build_lexical_identity_index(
        PROTOCOL / "littleprince_word_occurrences_v1.csv",
        split_index,
    )
    dataset = ContextEEGDataset(
        split_index=split_index,
        outer_fold=args.outer_fold,
        role="train",
        subject_index=subject_index,
        keyword_index=keyword_index,
        lexical_identity_index=lexical_index,
        sentence_labels_path=(
            PROTOCOL / "littleprince_sentence_keyword_labels_v1.csv"
        ),
        word_occurrences_path=(
            PROTOCOL / "littleprince_word_occurrences_v1.csv"
        ),
        text_backend=args.text_backend,
        context_store_root=CACHE_DIRS[args.text_backend],
        include_context_targets=True,
        expected_eeg_channels=128,
    )
    required = args.batch_size * args.num_batches
    indices = _scenario_indices(
        dataset,
        scenario=args.scenario,
        required=required,
    )
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=ContextEEGCollator(),
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        multiprocessing_context=(
            "spawn" if args.num_workers > 0 else None
        ),
    )

    eeg_model = build_eeg_sequence_encoder(
        args.eeg_config,
        actual_input_channels=128,
        actual_num_subjects=len(subject_index.subjects),
    ).to(device)
    text_projector = build_context_text_projector(
        TEXT_CONFIGS[args.text_backend],
        cache_metadata_path=(
            CACHE_DIRS[args.text_backend] / "metadata.json"
        ),
    ).to(device)
    sinkhorn_config = MaskedBalancedSinkhornConfig.from_yaml(
        DEFAULT_SINKHORN_CONFIG
    )
    transport_config = TransportPoolingConfig.from_yaml(
        DEFAULT_TRANSPORT_CONFIG
    )
    aligner = ContextOTAligner(
        text_projector=text_projector,
        sinkhorn=MaskedBalancedSinkhorn.from_config(sinkhorn_config),
        transport_eps=transport_config.eps,
    ).to(device)
    loss_module = ContextOTThreeLoss(
        ThreeLossConfig.from_yaml(args.loss_config)
    ).to(device)
    prototype_builder = TrainOnlyPrototypeBuilder(
        config=PrototypeBuilderConfig.from_yaml(args.prototype_config),
        split_index=split_index,
        outer_fold=args.outer_fold,
        keyword_index=keyword_index,
        lexical_identity_index=lexical_index,
        sentence_labels_path=(
            PROTOCOL / "littleprince_sentence_keyword_labels_v1.csv"
        ),
        word_occurrences_path=(
            PROTOCOL / "littleprince_word_occurrences_v1.csv"
        ),
        eligibility_path=(
            PROTOCOL / "littleprince_keyword_fold_eligibility_v1.csv"
        ),
        context_store_root=CACHE_DIRS[args.text_backend],
        text_backend=args.text_backend,
    )
    prototype_started = time.perf_counter()
    bank = prototype_builder.build(text_projector).to(device)
    prototype_seconds = time.perf_counter() - prototype_started

    batches = []
    for batch_index, batch in enumerate(loader):
        moved = batch.to(device, non_blocking=device.type == "cuda")
        if moved.context_words is None or moved.context_backend is None:
            raise RuntimeError("Train smoke requires contextual word targets")
        eeg_model.train()
        aligner.train()
        eeg_model.zero_grad(set_to_none=True)
        aligner.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        _synchronize(device)
        started = time.perf_counter()
        with _autocast(device, args.precision):
            eeg_output = eeg_model(
                eeg=moved.eeg,
                eeg_mask=moved.eeg_mask,
                subject_indices=moved.subject_indices,
            )
            alignment = aligner(
                eeg_sequence=eeg_output.sequence,
                eeg_mask=eeg_output.mask,
                context_words=moved.context_words,
                word_mask=moved.word_mask,
                context_backend=moved.context_backend,
            )
            losses = loss_module(
                alignment=alignment,
                word_mask=moved.word_mask,
                context_token_group_indices=(
                    moved.context_token_group_indices
                ),
                surface_type_indices=moved.surface_type_indices,
                sentence_group_indices=moved.sentence_group_indices,
                word_keyword_indices=moved.word_keyword_indices,
                prototype_bank=bank,
            )
        _synchronize(device)
        forward_seconds = time.perf_counter() - started
        started = time.perf_counter()
        losses.total.backward()
        _synchronize(device)
        backward_seconds = time.perf_counter() - started
        memory = (
            {
                "peak_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(device)
                ),
                "peak_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(device)
                ),
            }
            if device.type == "cuda"
            else None
        )
        batches.append(
            {
                "batch_index": batch_index,
                "eeg_input_shape": list(moved.eeg.shape),
                "eeg_input_lengths": moved.eeg_lengths.tolist(),
                "eeg_sequence_shape": list(eeg_output.sequence.shape),
                "eeg_output_lengths": eeg_output.lengths.tolist(),
                "context_input_shape": list(moved.context_words.shape),
                "projected_text_shape": list(
                    alignment.text.sequence.shape
                ),
                "word_lengths": moved.word_lengths.tolist(),
                "plan_shape": list(alignment.sinkhorn.plan.shape),
                "plan_dtype": str(alignment.sinkhorn.plan.dtype),
                "plan_total_mass": alignment.sinkhorn.plan.sum(
                    dim=(1, 2)
                ).detach().cpu().tolist(),
                "maximum_row_marginal_error": float(
                    alignment.sinkhorn.row_error.detach().max()
                ),
                "maximum_column_marginal_error": float(
                    alignment.sinkhorn.column_error.detach().max()
                ),
                "losses": {
                    "ot_context": float(losses.ot_context.detach()),
                    "context_token": float(losses.context_token.detach()),
                    "prototype": float(losses.prototype.detach()),
                    "total": float(losses.total.detach()),
                },
                "token_query_count": (
                    losses.token_output.valid_query_count
                ),
                "token_mean_positive_count": (
                    losses.diagnostics["token_mean_positive_count"]
                ),
                "token_false_negative_mask_count": (
                    losses.token_output.false_negative_mask_count
                ),
                "prototype_valid_word_count": (
                    losses.prototype_output.valid_word_count
                ),
                "scalar_mix_weights": (
                    losses.diagnostics["scalar_mix_weights"]
                ),
                "forward_seconds": forward_seconds,
                "backward_seconds": backward_seconds,
                "memory": memory,
                "eeg_gradients": _gradient_summary(eeg_model),
                "text_projector_gradients": _gradient_summary(
                    text_projector
                ),
                "all_finite": losses.diagnostics["all_finite"],
            }
        )

    properties: dict[str, Any] = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    if device.type == "cuda":
        gpu = torch.cuda.get_device_properties(device)
        properties.update(
            {
                "gpu_name": gpu.name,
                "gpu_total_memory_bytes": int(gpu.total_memory),
            }
        )
    result = {
        "outer_fold": args.outer_fold,
        "role": "train",
        "text_backend": args.text_backend,
        "precision": args.precision,
        "scenario": args.scenario,
        "batch_size": args.batch_size,
        "num_batches": args.num_batches,
        "train_dataset_size": len(dataset),
        "subject_adapter_enabled": (
            eeg_model.config.subject_adapter.enabled
        ),
        "prototype_bank": {
            "shape": list(bank.vectors.shape),
            "available_count": bank.available_count,
            "train_sentence_df_range": [
                int(bank.train_sentence_df.min()),
                int(bank.train_sentence_df.max()),
            ],
            "train_group_df_range": [
                int(bank.train_group_df.min()),
                int(bank.train_group_df.max()),
            ],
            "projector_state_hash": bank.projector_state_hash,
            "source_cache_hash": bank.source_cache_hash,
            "build_seconds": prototype_seconds,
        },
        "device": properties,
        "validation_test_context_accessed": False,
        "batches": batches,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
