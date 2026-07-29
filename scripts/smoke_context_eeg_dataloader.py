from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_keyword_decoding.data import (
    ContextEEGCollator,
    ContextEEGDataset,
    build_master_keyword_index,
    build_split_view_index,
    build_subject_index,
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


def _rss_bytes() -> int | None:
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.Process().memory_info().rss)


def _batch_summary(
    batch: Any,
    *,
    batch_index: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "batch_index": batch_index,
        "read_seconds": elapsed_seconds,
        "subjects": list(batch.subjects),
        "text_embedding_indices": batch.text_embedding_indices.tolist(),
        "eeg_lengths": batch.eeg_lengths.tolist(),
        "padded_eeg_shape": list(batch.eeg.shape),
        "word_lengths": batch.word_lengths.tolist(),
        "padded_context_shape": (
            list(batch.context_words.shape)
            if batch.context_words is not None
            else None
        ),
        "eeg_mask_valid_fraction": float(batch.eeg_mask.float().mean()),
        "word_mask_valid_fraction": float(batch.word_mask.float().mean()),
        "context_targets_read": batch.context_words is not None,
        "context_backend": batch.context_backend,
        "context_cache_metadata_sha256": (
            batch.context_cache_metadata_sha256
        ),
        "context_vectors_sha256": batch.context_vectors_sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read real variable EEG/context-word batches without training."
    )
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument(
        "--role",
        choices=("train", "validation", "test"),
        default="train",
    )
    parser.add_argument(
        "--text-backend",
        choices=tuple(CACHE_DIRS),
        default="bge_m3",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-batches", type=int, default=3)
    parser.add_argument("--pin-memory", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    if args.num_batches <= 0:
        raise ValueError("num-batches must be positive")

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
    include_context = args.role == "train"
    dataset = ContextEEGDataset(
        split_index=split_index,
        outer_fold=args.outer_fold,
        role=args.role,
        subject_index=subject_index,
        keyword_index=keyword_index,
        sentence_labels_path=(
            protocol / "littleprince_sentence_keyword_labels_v1.csv"
        ),
        word_occurrences_path=(
            protocol / "littleprince_word_occurrences_v1.csv"
        ),
        text_backend=args.text_backend,
        context_store_root=(
            CACHE_DIRS[args.text_backend] if include_context else None
        ),
        include_context_targets=include_context,
        expected_eeg_channels=128,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=ContextEEGCollator(),
        pin_memory=args.pin_memory,
        persistent_workers=False,
    )

    rss_start = _rss_bytes()
    peak_rss = rss_start
    tracemalloc.start()
    batches: list[dict[str, Any]] = []
    iterator = iter(loader)
    try:
        for batch_index in range(args.num_batches):
            started = time.perf_counter()
            batch = next(iterator)
            elapsed = time.perf_counter() - started
            batches.append(
                _batch_summary(
                    batch,
                    batch_index=batch_index,
                    elapsed_seconds=elapsed,
                )
            )
            current_rss = _rss_bytes()
            if current_rss is not None:
                peak_rss = max(peak_rss or current_rss, current_rss)
    except StopIteration as error:
        raise RuntimeError(
            f"Dataset ended before {args.num_batches} batches"
        ) from error
    finally:
        del iterator
    _, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    result = {
        "outer_fold": args.outer_fold,
        "role": args.role,
        "text_backend": args.text_backend,
        "dataset_samples": len(dataset),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "num_batches": len(batches),
        "pin_memory": args.pin_memory,
        "context_targets_allowed": include_context,
        "batches": batches,
        "main_process_rss_start_bytes": rss_start,
        "main_process_peak_observed_rss_bytes": peak_rss,
        "main_process_peak_observed_rss_delta_bytes": (
            peak_rss - rss_start
            if peak_rss is not None and rss_start is not None
            else None
        ),
        "python_tracemalloc_peak_bytes": traced_peak,
        "torch_version": torch.__version__,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
