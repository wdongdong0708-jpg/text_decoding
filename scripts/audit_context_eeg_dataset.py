from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_keyword_decoding.data import (
    build_master_keyword_index,
    build_split_view_index,
    build_subject_index,
)
from eeg_keyword_decoding.io import BrainVisionReader
from eeg_keyword_decoding.text import ContextWordStore, load_context_sentences


def _paths_by_recording(index: Any) -> dict[Path, list[Any]]:
    grouped: dict[Path, list[Any]] = defaultdict(list)
    for record in index.manifest_records:
        grouped[record.eeg_vhdr_path.resolve()].append(record)
    return dict(grouped)


def _audit_brainvision(index: Any) -> dict[str, Any]:
    recordings = _paths_by_recording(index)
    channel_counts: Counter[int] = Counter()
    sampling_frequencies: Counter[float] = Counter()
    for path, records in sorted(recordings.items(), key=lambda item: str(item[0])):
        reader = BrainVisionReader(path)
        channel_counts[reader.n_channels] += 1
        sampling_frequencies[reader.sfreq] += 1
        for record in records:
            if record.stop_sample > reader.n_samples:
                raise ValueError(
                    f"Manifest window exceeds recording for {record.eeg_view_id}"
                )
            if record.n_samples != record.stop_sample - record.start_sample:
                raise ValueError(
                    f"Manifest length mismatch for {record.eeg_view_id}"
                )
            if record.sfreq != reader.sfreq:
                raise ValueError(
                    f"Manifest sfreq mismatch for {record.eeg_view_id}"
                )
            if not record.events_tsv_path.is_file():
                raise FileNotFoundError(record.events_tsv_path)
        del reader
    gc.collect()
    if len(channel_counts) != 1:
        raise ValueError(
            f"BrainVision recordings have mixed channels: {channel_counts}"
        )
    return {
        "recording_count": len(recordings),
        "all_manifest_windows_in_bounds": True,
        "all_event_paths_accessible": True,
        "channel_counts_by_recording": {
            str(key): value for key, value in sorted(channel_counts.items())
        },
        "sampling_frequencies_by_recording": {
            str(key): value
            for key, value in sorted(sampling_frequencies.items())
        },
    }


def _audit_context_store(
    root: Path,
    sentences: list[Any],
) -> dict[str, Any]:
    store = ContextWordStore(root, verify_hashes=True)
    store.audit_against(sentences)
    descriptor = store.metadata.arrays["context_vectors"]
    return {
        "root": str(root.resolve()),
        "schema_version": store.metadata.schema_version,
        "shape": list(descriptor.shape),
        "dtype": descriptor.dtype,
        "context_vectors_sha256": descriptor.sha256,
        "model_revision": store.metadata.model["resolved_revision"],
        "tokenizer_revision": store.metadata.tokenizer[
            "resolved_revision"
        ],
        "frozen_order_matches": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the frozen split-to-variable-EEG Dataset contract."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "manifests"
            / "littleprince_pl_all_clean_manifest.csv"
        ),
    )
    parser.add_argument(
        "--protocol-dir",
        type=Path,
        default=(
            PROJECT_ROOT / "data" / "protocols" / "littleprince_hf_v1"
        ),
    )
    parser.add_argument(
        "--skip-brainvision",
        action="store_true",
        help="Skip local recording/header accessibility checks.",
    )
    args = parser.parse_args()

    protocol = args.protocol_dir.resolve()
    split_index = build_split_view_index(
        args.manifest.resolve(),
        protocol / "littleprince_sentence_folds_v1.csv",
    )
    subject_index = build_subject_index(split_index.manifest_records)
    keyword_index = build_master_keyword_index(
        protocol / "littleprince_hf_lexicon_v1.csv"
    )
    sentences = load_context_sentences(
        protocol / "littleprince_sentence_keyword_labels_v1.csv",
        protocol / "littleprince_word_occurrences_v1.csv",
    )
    if {sentence.text_embedding_idx for sentence in sentences} != set(
        split_index.valid_text_embedding_indices
    ):
        raise ValueError("Word occurrence and split sentence sets differ")

    views_by_sentence = Counter(
        record.text_embedding_idx
        for record in split_index.valid_manifest_records
    )
    result = {
        "outer_folds": list(split_index.outer_folds),
        "role_counts": split_index.role_counts(),
        "manifest_rows": len(split_index.manifest_records),
        "valid_sentences": len(split_index.valid_text_embedding_indices),
        "valid_eeg_views": split_index.valid_eeg_view_count,
        "excluded_sentence_indices": sorted(
            split_index.excluded_text_embedding_indices
        ),
        "excluded_eeg_views": split_index.excluded_eeg_view_count,
        "valid_views_per_sentence": {
            str(key): value
            for key, value in sorted(Counter(views_by_sentence.values()).items())
        },
        "minimum_views_per_valid_sentence": min(views_by_sentence.values()),
        "maximum_views_per_valid_sentence": max(views_by_sentence.values()),
        "word_occurrences": sum(
            sentence.word_count for sentence in sentences
        ),
        "maximum_words_per_sentence": max(
            sentence.word_count for sentence in sentences
        ),
        "subject_index": subject_index.to_metadata(),
        "keyword_index": keyword_index.to_metadata(),
        "context_stores": {
            "macbert": _audit_context_store(
                PROJECT_ROOT
                / "data"
                / "cache"
                / "context_words"
                / "macbert_v1",
                sentences,
            ),
            "bge_m3": _audit_context_store(
                PROJECT_ROOT
                / "data"
                / "cache"
                / "context_words"
                / "bge_m3_colbert_v1",
                sentences,
            ),
        },
        "brainvision": (
            None if args.skip_brainvision else _audit_brainvision(split_index)
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
