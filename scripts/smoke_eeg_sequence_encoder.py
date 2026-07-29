from __future__ import annotations

import argparse
import json
import math
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
from eeg_keyword_decoding.data.protocol_assets import file_sha256  # noqa: E402
from eeg_keyword_decoding.models import (  # noqa: E402
    build_eeg_sequence_encoder,
)


DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs" / "models" / "eeg_sequence_conv_v1.yaml"
)


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


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
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }


def _autocast(device: torch.device, precision: str):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=precision == "amp",
    )


def _valid_square_mean(
    sequence: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    valid = mask.unsqueeze(-1).expand_as(sequence)
    return sequence.float().square().masked_select(valid).mean()


def _benchmark_batch(
    *,
    model: torch.nn.Module,
    batch: Any,
    device: torch.device,
    precision: str,
) -> dict[str, Any]:
    eeg = batch.eeg.to(device, non_blocking=True)
    eeg_mask = batch.eeg_mask.to(device, non_blocking=True)
    subject_indices = batch.subject_indices.to(device, non_blocking=True)

    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode(), _autocast(device, precision):
        inference_output = model(
            eeg=eeg,
            eeg_mask=eeg_mask,
            subject_indices=subject_indices,
        )
    _synchronize(device)
    forward_seconds = time.perf_counter() - started
    forward_memory = _memory_snapshot(device)

    model.train()
    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    started = time.perf_counter()
    with _autocast(device, precision):
        output = model(
            eeg=eeg,
            eeg_mask=eeg_mask,
            subject_indices=subject_indices,
        )
        loss = _valid_square_mean(output.sequence, output.mask)
    loss.backward()
    _synchronize(device)
    forward_backward_seconds = time.perf_counter() - started
    forward_backward_memory = _memory_snapshot(device)

    invalid = output.sequence.masked_select(~output.mask.unsqueeze(-1))
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    result = {
        "input_eeg_shape": list(batch.eeg.shape),
        "input_eeg_dtype": str(batch.eeg.dtype),
        "input_lengths": batch.eeg_lengths.tolist(),
        "input_eeg_mask_dtype": str(batch.eeg_mask.dtype),
        "input_eeg_mask_is_prefix": bool(
            torch.equal(
                batch.eeg_mask.sum(dim=1),
                batch.eeg_lengths,
            )
        ),
        "subject_indices": batch.subject_indices.tolist(),
        "context_targets_read": batch.context_words is not None,
        "output_sequence_shape": list(output.sequence.shape),
        "output_dtype": str(output.sequence.dtype),
        "output_lengths": output.lengths.tolist(),
        "expected_output_lengths": [
            math.ceil(length / 4)
            for length in batch.eeg_lengths.tolist()
        ],
        "output_mask_dtype": str(output.mask.dtype),
        "output_valid_fraction": float(output.mask.float().mean()),
        "padding_region_max_abs_value": (
            float(invalid.abs().max().item()) if invalid.numel() else 0.0
        ),
        "finite_output": bool(torch.isfinite(output.sequence).all()),
        "finite_loss": bool(torch.isfinite(loss)),
        "finite_gradients": bool(
            gradients
            and all(torch.isfinite(gradient).all() for gradient in gradients)
        ),
        "virtual_loss": float(loss.detach()),
        "forward_seconds": forward_seconds,
        "forward_backward_seconds": forward_backward_seconds,
        "forward_memory": forward_memory,
        "forward_backward_memory": forward_backward_memory,
        "eval_train_output_shape_equal": (
            inference_output.sequence.shape == output.sequence.shape
        ),
    }
    model.zero_grad(set_to_none=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test the EEG sequence encoder on inner-train EEG."
    )
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--role", choices=("train",), default="train")
    parser.add_argument("--batch-size", type=int, default=8)
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
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--maximum-length-batch",
        action="store_true",
        help="Select the longest inner-train EEG views first.",
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
        include_context_targets=False,
        expected_eeg_channels=128,
    )
    selected: Any = dataset
    if args.maximum_length_batch:
        required = args.batch_size * args.num_batches
        indices = sorted(
            range(len(dataset.records)),
            key=lambda index: dataset.records[index].n_samples,
            reverse=True,
        )[:required]
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
    model = build_eeg_sequence_encoder(
        args.config.resolve(),
        actual_input_channels=128,
        actual_num_subjects=len(subject_index.subjects),
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
            result = _benchmark_batch(
                model=model,
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
                "compute_capability": list(
                    torch.cuda.get_device_capability(device)
                ),
            }
        )
    result = {
        "outer_fold": args.outer_fold,
        "role": args.role,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "num_batches": args.num_batches,
        "precision": args.precision,
        "maximum_length_batch": args.maximum_length_batch,
        "dataset_samples": len(dataset),
        "text_context_targets_deliberately_disabled": True,
        "model": {
            "config_path": str(args.config.resolve()),
            "config_file_sha256": file_sha256(args.config.resolve()),
            "canonical_config_sha256": model.config.canonical_sha256,
            "total_parameters": model.total_parameter_count,
            "trainable_parameters": model.trainable_parameter_count,
            "total_stride": model.config.total_stride,
            "output_dim": model.config.output_dim,
            "subject_adapter_enabled": (
                model.config.subject_adapter.enabled
            ),
        },
        "device": device_metadata,
        "batches": batches,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
