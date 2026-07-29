from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_keyword_decoding.data import (  # noqa: E402
    ContextEEGCollator,
    ContextEEGDataset,
    build_master_keyword_index,
    build_split_view_index,
    build_subject_index,
)
from eeg_keyword_decoding.models import (  # noqa: E402
    build_context_text_projector,
    build_eeg_sequence_encoder,
)
from eeg_keyword_decoding.ot import (  # noqa: E402
    ContextOTAligner,
    MaskedBalancedSinkhorn,
    MaskedBalancedSinkhornConfig,
    TransportPoolingConfig,
)


CACHE_DIRS = {
    "macbert": (
        PROJECT_ROOT / "data" / "cache" / "context_words" / "macbert_v1"
    ),
    "bge_m3": (
        PROJECT_ROOT
        / "data"
        / "cache"
        / "context_words"
        / "bge_m3_colbert_v1"
    ),
}
TEXT_CONFIGS = {
    "macbert": (
        PROJECT_ROOT / "configs" / "text" / "macbert_projection_v1.yaml"
    ),
    "bge_m3": (
        PROJECT_ROOT / "configs" / "text" / "bge_m3_projection_v1.yaml"
    ),
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


def _memory_snapshot(device: torch.device) -> dict[str, int] | None:
    if device.type != "cuda":
        return None
    return {
        "peak_allocated_bytes": int(
            torch.cuda.max_memory_allocated(device)
        ),
        "peak_reserved_bytes": int(
            torch.cuda.max_memory_reserved(device)
        ),
    }


def _benchmark(
    *,
    eeg_model: torch.nn.Module,
    aligner: ContextOTAligner,
    batch: Any,
    device: torch.device,
    precision: str,
) -> dict[str, Any]:
    moved = batch.to(device, non_blocking=device.type == "cuda")
    if moved.context_words is None or moved.context_backend is None:
        raise RuntimeError("OT smoke requires inner-train context targets")

    eeg_model.eval()
    aligner.eval()
    _synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode(), _autocast(device, precision):
        eeg_output = eeg_model(
            eeg=moved.eeg,
            eeg_mask=moved.eeg_mask,
            subject_indices=moved.subject_indices,
        )
        inference = aligner(
            eeg_sequence=eeg_output.sequence,
            eeg_mask=eeg_output.mask,
            context_words=moved.context_words,
            word_mask=moved.word_mask,
            context_backend=moved.context_backend,
        )
    _synchronize(device)
    inference_forward_seconds = time.perf_counter() - started

    eeg_model.train()
    aligner.train()
    eeg_model.zero_grad(set_to_none=True)
    aligner.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    started = time.perf_counter()
    with _autocast(device, precision):
        eeg_output = eeg_model(
            eeg=moved.eeg,
            eeg_mask=moved.eeg_mask,
            subject_indices=moved.subject_indices,
        )
        output = aligner(
            eeg_sequence=eeg_output.sequence,
            eeg_mask=eeg_output.mask,
            context_words=moved.context_words,
            word_mask=moved.word_mask,
            context_backend=moved.context_backend,
        )
        audit_scalar = output.sinkhorn.transport_cost.mean()
    _synchronize(device)
    forward_seconds = time.perf_counter() - started
    started = time.perf_counter()
    audit_scalar.backward()
    _synchronize(device)
    backward_seconds = time.perf_counter() - started
    forward_backward_seconds = forward_seconds + backward_seconds
    memory = _memory_snapshot(device)

    pair_mask = output.cost.valid_pair_mask
    padding_plan = output.sinkhorn.plan[~pair_mask]
    gradients = [
        parameter.grad
        for parameter in (
            list(eeg_model.parameters())
            + list(aligner.parameters())
        )
        if parameter.requires_grad and parameter.grad is not None
    ]
    scalar_mix = output.text.scalar_mix_weights
    result = {
        "eeg_input_shape": list(moved.eeg.shape),
        "eeg_sequence_shape": list(eeg_output.sequence.shape),
        "eeg_input_lengths": moved.eeg_lengths.tolist(),
        "eeg_output_lengths": eeg_output.lengths.tolist(),
        "eeg_mask_dtype": str(eeg_output.mask.dtype),
        "text_backend": moved.context_backend,
        "context_input_shape": list(moved.context_words.shape),
        "projected_text_shape": list(output.text.sequence.shape),
        "word_lengths": moved.word_lengths.tolist(),
        "word_mask_dtype": str(moved.word_mask.dtype),
        "context_cache_metadata_sha256": (
            moved.context_cache_metadata_sha256
        ),
        "context_vectors_sha256": moved.context_vectors_sha256,
        "cost_matrix_shape": list(output.cost.cost.shape),
        "cost_dtype": str(output.cost.cost.dtype),
        "cost_range": [
            float(output.cost.cost.detach()[pair_mask].min()),
            float(output.cost.cost.detach()[pair_mask].max()),
        ],
        "plan_shape": list(output.sinkhorn.plan.shape),
        "plan_dtype": str(output.sinkhorn.plan.dtype),
        "plan_total_mass": (
            output.sinkhorn.plan.sum(dim=(1, 2)).tolist()
        ),
        "maximum_row_marginal_error": float(
            output.sinkhorn.row_error.detach().max()
        ),
        "maximum_column_marginal_error": float(
            output.sinkhorn.column_error.detach().max()
        ),
        "padding_plan_max_abs_value": (
            float(padding_plan.detach().abs().max())
            if padding_plan.numel()
            else 0.0
        ),
        "transport_cost_range": [
            float(output.sinkhorn.transport_cost.detach().min()),
            float(output.sinkhorn.transport_cost.detach().max()),
        ],
        "entropy_range": [
            float(output.sinkhorn.entropy.detach().min()),
            float(output.sinkhorn.entropy.detach().max()),
        ],
        "word_conditioned_eeg_shape": list(
            output.word_conditioned_eeg.sequence.shape
        ),
        "column_mass_range": [
            float(
                output.word_conditioned_eeg.column_mass.detach()[
                    moved.word_mask
                ].min()
            ),
            float(
                output.word_conditioned_eeg.column_mass.detach()[
                    moved.word_mask
                ].max()
            ),
        ],
        "scalar_mix_weights": (
            scalar_mix.detach().float().tolist()
            if scalar_mix is not None
            else None
        ),
        "inference_forward_seconds": inference_forward_seconds,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "forward_backward_seconds": forward_backward_seconds,
        "memory": memory,
        "audit_scalar": float(audit_scalar.detach()),
        "finite_outputs": all(
            bool(torch.isfinite(value).all())
            for value in (
                output.text.sequence,
                output.cost.cost,
                output.sinkhorn.plan,
                output.sinkhorn.transport_cost,
                output.sinkhorn.entropy,
                output.word_conditioned_eeg.sequence,
            )
        ),
        "finite_gradients": bool(
            gradients
            and all(
                torch.isfinite(gradient).all()
                for gradient in gradients
            )
        ),
        "inference_training_shape_equal": (
            inference.sinkhorn.plan.shape == output.sinkhorn.plan.shape
        ),
    }
    eeg_model.zero_grad(set_to_none=True)
    aligner.zero_grad(set_to_none=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test text projection and balanced context OT."
    )
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--role", choices=("train",), default="train")
    parser.add_argument(
        "--text-backend",
        choices=tuple(TEXT_CONFIGS),
        default="bge_m3",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-batches", type=int, default=3)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "amp"),
        default="fp32",
    )
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--iterations", type=int)
    parser.add_argument(
        "--eeg-config",
        type=Path,
        default=DEFAULT_EEG_CONFIG,
    )
    parser.add_argument(
        "--maximum-length-batch",
        action="store_true",
    )
    args = parser.parse_args()
    if args.batch_size <= 0 or args.num_batches <= 0:
        raise ValueError("batch-size and num-batches must be positive")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    device = _resolve_device(args.device)
    if args.precision == "amp" and device.type != "cuda":
        raise ValueError("AMP smoke requires CUDA")

    protocol = (
        PROJECT_ROOT / "data" / "protocols" / "littleprince_hf_v1"
    )
    split_index = build_split_view_index(
        PROJECT_ROOT
        / "data"
        / "manifests"
        / "littleprince_pl_all_clean_manifest.csv",
        protocol / "littleprince_sentence_folds_v1.csv",
    )
    subject_index = build_subject_index(split_index.manifest_records)
    keyword_index = build_master_keyword_index(
        protocol / "littleprince_hf_lexicon_v1.csv"
    )
    dataset = ContextEEGDataset(
        split_index=split_index,
        outer_fold=args.outer_fold,
        role="train",
        subject_index=subject_index,
        keyword_index=keyword_index,
        sentence_labels_path=(
            protocol / "littleprince_sentence_keyword_labels_v1.csv"
        ),
        word_occurrences_path=(
            protocol / "littleprince_word_occurrences_v1.csv"
        ),
        text_backend=args.text_backend,
        context_store_root=CACHE_DIRS[args.text_backend],
        include_context_targets=True,
        expected_eeg_channels=128,
    )
    selected: Any = dataset
    if args.maximum_length_batch:
        required = args.batch_size * args.num_batches
        word_count_by_sentence = {
            sentence.text_embedding_idx: sentence.word_count
            for sentence in dataset.sentences
        }
        eeg_ranked = sorted(
            range(len(dataset.records)),
            key=lambda index: dataset.records[index].n_samples,
            reverse=True,
        )
        word_ranked = sorted(
            range(len(dataset.records)),
            key=lambda index: word_count_by_sentence[
                dataset.records[index].text_embedding_idx
            ],
            reverse=True,
        )
        indices: list[int] = []
        for candidate in (eeg_ranked[0], word_ranked[0]):
            if len(indices) == required:
                break
            if candidate not in indices:
                indices.append(candidate)
        joint_ranked = sorted(
            range(len(dataset.records)),
            key=lambda index: (
                dataset.records[index].n_samples
                * word_count_by_sentence[
                    dataset.records[index].text_embedding_idx
                ],
                dataset.records[index].n_samples,
                word_count_by_sentence[
                    dataset.records[index].text_embedding_idx
                ],
            ),
            reverse=True,
        )
        for candidate in joint_ranked:
            if candidate not in indices:
                indices.append(candidate)
            if len(indices) == required:
                break
        selected = Subset(dataset, indices)
    loader = DataLoader(
        selected,
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
        args.eeg_config.resolve(),
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
    sinkhorn = MaskedBalancedSinkhorn(
        epsilon=(
            args.epsilon
            if args.epsilon is not None
            else sinkhorn_config.epsilon
        ),
        iterations=(
            args.iterations
            if args.iterations is not None
            else sinkhorn_config.iterations
        ),
    )
    transport_config = TransportPoolingConfig.from_yaml(
        DEFAULT_TRANSPORT_CONFIG
    )
    aligner = ContextOTAligner(
        text_projector=text_projector,
        sinkhorn=sinkhorn,
        transport_eps=transport_config.eps,
    ).to(device)

    batches: list[dict[str, Any]] = []
    iterator = iter(loader)
    try:
        for batch_index in range(args.num_batches):
            try:
                batch = next(iterator)
            except StopIteration as error:
                raise RuntimeError(
                    f"Dataset ended before batch {batch_index}"
                ) from error
            result = _benchmark(
                eeg_model=eeg_model,
                aligner=aligner,
                batch=batch,
                device=device,
                precision=args.precision,
            )
            result["batch_index"] = batch_index
            batches.append(result)
    finally:
        del iterator

    device_metadata: dict[str, Any] = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        device_metadata.update(
            {
                "gpu_name": properties.name,
                "gpu_total_memory_bytes": int(properties.total_memory),
            }
        )
    result = {
        "outer_fold": args.outer_fold,
        "role": "train",
        "text_backend": args.text_backend,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "num_batches": args.num_batches,
        "precision": args.precision,
        "maximum_length_batch": args.maximum_length_batch,
        "dataset_samples": len(dataset),
        "validation_test_context_accessed": False,
        "model": {
            "eeg_config": str(args.eeg_config.resolve()),
            "subject_adapter_enabled": (
                eeg_model.config.subject_adapter.enabled
            ),
            "eeg_parameters": eeg_model.total_parameter_count,
            "text_projection_parameters": (
                text_projector.total_parameter_count
            ),
            "sinkhorn_parameters": sum(
                parameter.numel()
                for parameter in sinkhorn.parameters()
            ),
            "epsilon": sinkhorn.epsilon,
            "iterations": sinkhorn.iterations,
            "internal_dtype": sinkhorn_config.internal_dtype,
        },
        "device": device_metadata,
        "batches": batches,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
