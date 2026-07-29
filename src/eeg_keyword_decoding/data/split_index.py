from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .eeg_manifest import (
    EEGManifestRecord,
    load_eeg_manifest,
    validate_eeg_manifest,
)


SplitRole = Literal["train", "validation", "test"]
SPLIT_ROLES: tuple[SplitRole, ...] = ("train", "validation", "test")


@dataclass(frozen=True)
class SentenceFoldAssignment:
    outer_fold: int
    role: SplitRole
    text_embedding_idx: int
    sentence_group_id: str
    normalized_text_sha256: str
    token_count: int
    protocol_version: str
    seed: int


@dataclass(frozen=True)
class SplitViewIndex:
    """Read-only join between frozen sentence folds and EEG manifest views."""

    manifest_records: tuple[EEGManifestRecord, ...]
    valid_manifest_records: tuple[EEGManifestRecord, ...]
    excluded_manifest_records: tuple[EEGManifestRecord, ...]
    assignments: dict[tuple[int, int], SentenceFoldAssignment]
    outer_folds: tuple[int, ...]
    valid_text_embedding_indices: frozenset[int]
    excluded_text_embedding_indices: frozenset[int]

    def assignment(
        self,
        outer_fold: int,
        text_embedding_idx: int,
    ) -> SentenceFoldAssignment:
        try:
            return self.assignments[(int(outer_fold), int(text_embedding_idx))]
        except KeyError as error:
            raise KeyError(
                "No frozen fold assignment for "
                f"outer_fold={outer_fold}, "
                f"text_embedding_idx={text_embedding_idx}"
            ) from error

    def records_for(
        self,
        outer_fold: int,
        role: SplitRole,
    ) -> tuple[EEGManifestRecord, ...]:
        _validate_fold_and_role(self, outer_fold, role)
        return tuple(
            record
            for record in self.valid_manifest_records
            if self.assignment(outer_fold, record.text_embedding_idx).role
            == role
        )

    def sentence_indices_for(
        self,
        outer_fold: int,
        role: SplitRole,
    ) -> tuple[int, ...]:
        _validate_fold_and_role(self, outer_fold, role)
        return tuple(
            sorted(
                assignment.text_embedding_idx
                for assignment in self.assignments.values()
                if assignment.outer_fold == outer_fold
                and assignment.role == role
            )
        )

    def role_counts(self) -> dict[int, dict[str, dict[str, int]]]:
        summary: dict[int, dict[str, dict[str, int]]] = {}
        for outer_fold in self.outer_folds:
            summary[outer_fold] = {}
            for role in SPLIT_ROLES:
                summary[outer_fold][role] = {
                    "sentences": len(
                        self.sentence_indices_for(outer_fold, role)
                    ),
                    "eeg_views": len(self.records_for(outer_fold, role)),
                }
        return summary

    @property
    def valid_eeg_view_count(self) -> int:
        return len(self.valid_manifest_records)

    @property
    def excluded_eeg_view_count(self) -> int:
        return len(self.excluded_manifest_records)


def _validate_fold_and_role(
    index: SplitViewIndex,
    outer_fold: int,
    role: str,
) -> None:
    if int(outer_fold) not in index.outer_folds:
        raise ValueError(
            f"outer_fold must be one of {index.outer_folds}, got {outer_fold}"
        )
    if role not in SPLIT_ROLES:
        raise ValueError(f"Unknown split role: {role!r}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def _load_fold_assignments(
    fold_path: Path,
) -> tuple[
    dict[tuple[int, int], SentenceFoldAssignment],
    tuple[int, ...],
    frozenset[int],
]:
    assignments: dict[tuple[int, int], SentenceFoldAssignment] = {}
    folds_by_sentence: dict[int, set[int]] = defaultdict(set)
    roles_by_fold_group: dict[tuple[int, str], set[str]] = defaultdict(set)
    group_by_sentence: dict[int, str] = {}
    test_counts: Counter[int] = Counter()

    for row in _read_csv(fold_path):
        role = row["role"]
        if role not in SPLIT_ROLES:
            raise ValueError(f"Unknown role in frozen fold file: {role!r}")
        outer_fold = int(row["outer_fold"])
        text_embedding_idx = int(row["text_embedding_idx"])
        group_id = row["sentence_group_id"]
        key = (outer_fold, text_embedding_idx)
        if key in assignments:
            raise ValueError(f"Duplicate fold assignment: {key}")
        previous_group = group_by_sentence.setdefault(
            text_embedding_idx,
            group_id,
        )
        if previous_group != group_id:
            raise ValueError(
                "sentence_group_id changed across outer folds for "
                f"text_embedding_idx={text_embedding_idx}"
            )
        assignment = SentenceFoldAssignment(
            outer_fold=outer_fold,
            role=role,  # type: ignore[arg-type]
            text_embedding_idx=text_embedding_idx,
            sentence_group_id=group_id,
            normalized_text_sha256=row["normalized_text_sha256"],
            token_count=int(row["token_count"]),
            protocol_version=row["protocol_version"],
            seed=int(row["seed"]),
        )
        assignments[key] = assignment
        folds_by_sentence[text_embedding_idx].add(outer_fold)
        roles_by_fold_group[(outer_fold, group_id)].add(role)
        if role == "test":
            test_counts[text_embedding_idx] += 1

    outer_folds = tuple(sorted({key[0] for key in assignments}))
    if outer_folds != tuple(range(len(outer_folds))):
        raise ValueError(
            f"Outer folds must be contiguous from zero, got {outer_folds}"
        )
    expected_folds = set(outer_folds)
    incomplete = {
        text_embedding_idx: sorted(folds)
        for text_embedding_idx, folds in folds_by_sentence.items()
        if folds != expected_folds
    }
    if incomplete:
        raise ValueError(
            "Every valid sentence must have one assignment in every fold: "
            f"{list(incomplete.items())[:5]}"
        )
    leaking_groups = [
        key for key, roles in roles_by_fold_group.items() if len(roles) != 1
    ]
    if leaking_groups:
        raise ValueError(
            "Normalized sentence groups cross roles: "
            f"{leaking_groups[:5]}"
        )
    if set(test_counts) != set(folds_by_sentence):
        raise ValueError("Some valid sentences never appear in outer-test")
    if set(test_counts.values()) != {1}:
        raise ValueError(
            "Every valid sentence must appear in outer-test exactly once"
        )
    return assignments, outer_folds, frozenset(folds_by_sentence)


def build_split_view_index(
    manifest_path: str | Path,
    fold_path: str | Path,
    *,
    minimum_views_per_sentence: int = 6,
    maximum_views_per_sentence: int = 8,
) -> SplitViewIndex:
    """Build and fully audit the frozen fold-to-EEG-view join.

    This function reads manifest metadata only. It does not open or read EEG
    recordings.
    """

    manifest_records = tuple(load_eeg_manifest(manifest_path))
    validate_eeg_manifest(list(manifest_records), check_files=False)
    assignments, outer_folds, valid_indices = _load_fold_assignments(
        Path(fold_path)
    )
    records_by_sentence: dict[int, list[EEGManifestRecord]] = defaultdict(list)
    for record in manifest_records:
        records_by_sentence[record.text_embedding_idx].append(record)

    missing_manifest = valid_indices - set(records_by_sentence)
    if missing_manifest:
        raise ValueError(
            "Frozen fold sentences are missing from the EEG manifest: "
            f"{sorted(missing_manifest)[:10]}"
        )
    view_count_failures = {
        text_embedding_idx: len(records_by_sentence[text_embedding_idx])
        for text_embedding_idx in valid_indices
        if not (
            minimum_views_per_sentence
            <= len(records_by_sentence[text_embedding_idx])
            <= maximum_views_per_sentence
        )
    }
    if view_count_failures:
        raise ValueError(
            "Unexpected valid EEG view count per sentence: "
            f"{list(view_count_failures.items())[:10]}"
        )

    valid_records = tuple(
        record
        for record in manifest_records
        if record.text_embedding_idx in valid_indices
    )
    excluded_records = tuple(
        record
        for record in manifest_records
        if record.text_embedding_idx not in valid_indices
    )
    excluded_indices = frozenset(
        record.text_embedding_idx for record in excluded_records
    )
    index = SplitViewIndex(
        manifest_records=manifest_records,
        valid_manifest_records=valid_records,
        excluded_manifest_records=excluded_records,
        assignments=assignments,
        outer_folds=outer_folds,
        valid_text_embedding_indices=valid_indices,
        excluded_text_embedding_indices=excluded_indices,
    )

    for outer_fold in outer_folds:
        seen_view_ids: set[str] = set()
        for role in SPLIT_ROLES:
            for record in index.records_for(outer_fold, role):
                if record.eeg_view_id in seen_view_ids:
                    raise ValueError(
                        f"EEG view crosses roles in fold {outer_fold}: "
                        f"{record.eeg_view_id}"
                    )
                seen_view_ids.add(record.eeg_view_id)
        expected_view_ids = {
            record.eeg_view_id for record in valid_records
        }
        if seen_view_ids != expected_view_ids:
            raise ValueError(
                f"Fold {outer_fold} does not cover every valid EEG view"
            )
    return index
