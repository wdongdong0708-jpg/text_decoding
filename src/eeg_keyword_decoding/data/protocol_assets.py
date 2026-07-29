from __future__ import annotations

import csv
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any

from .eeg_manifest import group_records_by_sentence, load_eeg_manifest, validate_eeg_manifest


class ProtocolAuditError(ValueError):
    """Raised when a frozen protocol asset violates its v1 contract."""


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ProtocolAuditError(f"CSV is empty: {path}")
    return rows


def _pipe_values(value: str) -> list[str]:
    return [part for part in value.split("|") if part]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolAuditError(message)


def audit_littleprince_hf_v1(
    protocol_dir: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    protocol_root = Path(protocol_dir)
    lexicon_path = protocol_root / "littleprince_hf_lexicon_v1.csv"
    labels_path = protocol_root / "littleprince_sentence_keyword_labels_v1.csv"

    lexicon = _read_csv(lexicon_path)
    labels = _read_csv(labels_path)
    records = load_eeg_manifest(manifest_path)
    validate_eeg_manifest(records, check_files=False)

    keyword_ids = [row["keyword_id"] for row in lexicon]
    _require(len(keyword_ids) == len(set(keyword_ids)), "Duplicate keyword_id")

    core = {row["keyword_id"] for row in lexicon if row["include_core"] == "true"}
    main = {row["keyword_id"] for row in lexicon if row["include_main"] == "true"}
    extended = {
        row["keyword_id"] for row in lexicon if row["include_extended"] == "true"
    }
    master = set(keyword_ids)

    _require(core <= main <= extended <= master, "Lexicon tiers are not nested")
    _require(len(master) == 247, f"Expected 247 Master words, got {len(master)}")
    _require(len(core) == 33, f"Expected 33 Core words, got {len(core)}")
    _require(len(main) == 64, f"Expected 64 Main words, got {len(main)}")
    _require(
        len(extended) == 100,
        f"Expected 100 Extended words, got {len(extended)}",
    )

    sentence_indices = [int(row["text_embedding_idx"]) for row in labels]
    _require(
        len(sentence_indices) == len(set(sentence_indices)),
        "Duplicate text_embedding_idx in sentence labels",
    )
    _require(
        sentence_indices == list(range(16, 2853)),
        "Sentence label indices must be continuous from 16 through 2852",
    )

    unknown_references: set[str] = set()
    total_word_occurrences = 0
    valid_sentence_count = 0
    chapter_heading_count = 0
    empty_master_count = 0
    for row in labels:
        total_word_occurrences += int(row["token_count"])
        is_heading = row["is_chapter_heading"] == "true"
        chapter_heading_count += int(is_heading)
        valid_sentence_count += int(not is_heading and int(row["token_count"]) > 0)
        empty_master_count += int(int(row["keyword_count_all"]) == 0)
        for field in (
            "present_keyword_ids",
            "core_keyword_ids",
            "main_keyword_ids",
            "extended_keyword_ids",
        ):
            unknown_references.update(set(_pipe_values(row[field])) - master)

    _require(not unknown_references, f"Unknown keyword references: {unknown_references}")
    _require(
        total_word_occurrences == 14_034,
        f"Expected 14034 word occurrences, got {total_word_occurrences}",
    )
    _require(
        valid_sentence_count == 2_809,
        f"Expected 2809 valid word sequences, got {valid_sentence_count}",
    )
    _require(
        chapter_heading_count == 27,
        f"Expected 27 chapter headings, got {chapter_heading_count}",
    )

    grouped_records = group_records_by_sentence(records)
    manifest_indices = set(grouped_records)
    _require(
        manifest_indices == set(sentence_indices),
        "EEG manifest and sentence labels cover different text_embedding_idx values",
    )
    view_counts = Counter(len(views) for views in grouped_records.values())
    _require(
        set(view_counts) <= {6, 7, 8},
        f"Unexpected number of EEG views per sentence: {dict(view_counts)}",
    )

    story_local_count = sum(
        row["story_local_flag"] == "true" for row in lexicon
    )
    return {
        "master_words": len(master),
        "core_words": len(core),
        "main_words": len(main),
        "extended_words": len(extended),
        "story_local_words": story_local_count,
        "sentence_rows": len(labels),
        "valid_word_sequences": valid_sentence_count,
        "chapter_headings": chapter_heading_count,
        "empty_master_sentences": empty_master_count,
        "word_occurrences": total_word_occurrences,
        "eeg_rows": len(records),
        "eeg_view_counts": dict(sorted(view_counts.items())),
        "lexicon_sha256": file_sha256(lexicon_path),
        "labels_sha256": file_sha256(labels_path),
        "manifest_sha256": file_sha256(manifest_path),
    }

