from __future__ import annotations

import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .protocol_assets import file_sha256
from .word_occurrences import normalized_text_sha256


OCCURRENCE_LABEL_WEIGHT = 0.1
CONTEXT_GROUP_LABEL_WEIGHT = 0.9
SIZE_WEIGHT = 0.25


FOLD_FIELDS = (
    "protocol_version",
    "seed",
    "outer_fold",
    "role",
    "text_embedding_idx",
    "sentence_group_id",
    "normalized_text_sha256",
    "chapter",
    "story_quartile",
    "token_count",
)

ELIGIBILITY_FIELDS = (
    "outer_fold",
    "keyword_id",
    "surface_form",
    "include_core",
    "include_main",
    "include_extended",
    "story_local_flag",
    "train_df",
    "validation_df",
    "test_df",
    "train_group_df",
    "validation_group_df",
    "test_group_df",
    "eligible_1_1_1",
    "eligible_10_3_3",
    "eligible_20_5_5",
    "eligible_group_1_1_1",
    "eligible_group_10_3_3",
    "eligible_group_20_5_5",
)


@dataclass(frozen=True)
class SentenceExample:
    text_embedding_idx: int
    normalized_text_hash: str
    chapter: int
    story_quartile: int
    token_count: int
    labels: frozenset[str]

    @property
    def group_id(self) -> str:
        return f"lptext:{self.normalized_text_hash}"


@dataclass(frozen=True)
class SentenceGroup:
    group_id: str
    examples: tuple[SentenceExample, ...]
    label_counts: Counter[str]

    @property
    def size(self) -> int:
        return len(self.examples)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def load_sentence_examples(
    sentence_labels_path: str | Path,
    *,
    label_field: str = "main_keyword_ids",
) -> list[SentenceExample]:
    examples: list[SentenceExample] = []
    for row in _read_csv(Path(sentence_labels_path)):
        if row["is_chapter_heading"] == "true" or int(row["token_count"]) <= 0:
            continue
        labels = frozenset(value for value in row[label_field].split("|") if value)
        examples.append(
            SentenceExample(
                text_embedding_idx=int(row["text_embedding_idx"]),
                normalized_text_hash=normalized_text_sha256(row["text"]),
                chapter=int(row["chapter"]),
                story_quartile=int(row["story_quartile"]),
                token_count=int(row["token_count"]),
                labels=labels,
            )
        )
    return examples


def group_sentence_examples(
    examples: Iterable[SentenceExample],
) -> list[SentenceGroup]:
    grouped: dict[str, list[SentenceExample]] = defaultdict(list)
    for example in examples:
        grouped[example.group_id].append(example)

    output = []
    for group_id, members in grouped.items():
        label_counts: Counter[str] = Counter()
        for member in members:
            label_counts.update(sorted(member.labels))
        output.append(
            SentenceGroup(
                group_id=group_id,
                examples=tuple(
                    sorted(members, key=lambda item: item.text_embedding_idx)
                ),
                label_counts=label_counts,
            )
        )
    return sorted(output, key=lambda group: group.group_id)


def _assignment_objective(
    fold_sizes: list[int],
    fold_label_counts: list[Counter[str]],
    fold_group_label_counts: list[Counter[str]],
    total_size: int,
    total_label_counts: Counter[str],
    total_group_label_counts: Counter[str],
) -> float:
    n_splits = len(fold_sizes)
    target_size = total_size / n_splits
    size_cost = sum(
        ((size - target_size) / max(target_size, 1.0)) ** 2
        for size in fold_sizes
    )
    if not total_label_counts:
        return size_cost

    occurrence_label_cost = 0.0
    context_group_label_cost = 0.0
    for label, total in sorted(total_label_counts.items()):
        target = total / n_splits
        denominator = max(target, 1.0)
        occurrence_label_cost += sum(
            ((counts[label] - target) / denominator) ** 2
            for counts in fold_label_counts
        )
        group_target = total_group_label_counts[label] / n_splits
        group_denominator = max(group_target, 1.0)
        context_group_label_cost += sum(
            ((counts[label] - group_target) / group_denominator) ** 2
            for counts in fold_group_label_counts
        )
    occurrence_label_cost /= len(total_label_counts)
    context_group_label_cost /= len(total_label_counts)
    return (
        OCCURRENCE_LABEL_WEIGHT * occurrence_label_cost
        + CONTEXT_GROUP_LABEL_WEIGHT * context_group_label_cost
        + SIZE_WEIGHT * size_cost
    )


def assign_groups_multilabel(
    groups: list[SentenceGroup],
    *,
    n_splits: int,
    seed: int,
    restarts: int = 32,
) -> dict[str, int]:
    """Assign atomic sentence groups with deterministic multilabel balancing.

    Groups containing rarer labels are placed first. Each placement minimizes
    the incremental squared deviation from equal sentence counts, per-label
    sentence-occurrence counts, and per-label distinct-context-group counts.
    Multiple deterministic fold-order restarts are evaluated, and the globally
    best assignment under the same objective is returned.
    """

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if len(groups) < n_splits:
        raise ValueError("Cannot create more folds than sentence groups")
    if restarts <= 0:
        raise ValueError("restarts must be positive")

    total_size = sum(group.size for group in groups)
    total_label_counts: Counter[str] = Counter()
    total_group_label_counts: Counter[str] = Counter()
    for group in groups:
        for label, count in sorted(group.label_counts.items()):
            total_label_counts[label] += count
            total_group_label_counts[label] += 1

    def priority(group: SentenceGroup) -> tuple[float, int, str]:
        occurrence_rarity = sum(
            count / max(total_label_counts[label], 1)
            for label, count in sorted(group.label_counts.items())
        )
        context_group_rarity = sum(
            1 / max(total_group_label_counts[label], 1)
            for label in sorted(group.label_counts)
        )
        rarity = (
            OCCURRENCE_LABEL_WEIGHT * occurrence_rarity
            + CONTEXT_GROUP_LABEL_WEIGHT * context_group_rarity
        )
        return (-rarity, -group.size, group.group_id)

    ordered_groups = sorted(groups, key=priority)
    target_size = total_size / n_splits
    label_count = max(len(total_label_counts), 1)
    best_assignment: dict[str, int] | None = None
    best_objective = math.inf

    for restart in range(restarts):
        fold_sizes = [0] * n_splits
        fold_label_counts = [Counter() for _ in range(n_splits)]
        fold_group_label_counts = [Counter() for _ in range(n_splits)]
        assignment: dict[str, int] = {}
        rng = random.Random(seed + restart * 1_000_003)

        for group in ordered_groups:
            fold_order = list(range(n_splits))
            rng.shuffle(fold_order)
            best_fold = -1
            best_delta = math.inf
            for fold in fold_order:
                old_size_error = (
                    (fold_sizes[fold] - target_size) / max(target_size, 1.0)
                ) ** 2
                new_size_error = (
                    (fold_sizes[fold] + group.size - target_size)
                    / max(target_size, 1.0)
                ) ** 2
                delta = SIZE_WEIGHT * (new_size_error - old_size_error)

                for label, increment in sorted(group.label_counts.items()):
                    target = total_label_counts[label] / n_splits
                    denominator = max(target, 1.0)
                    old_error = (
                        (fold_label_counts[fold][label] - target) / denominator
                    ) ** 2
                    new_error = (
                        (
                            fold_label_counts[fold][label]
                            + increment
                            - target
                        )
                        / denominator
                    ) ** 2
                    group_target = total_group_label_counts[label] / n_splits
                    group_denominator = max(group_target, 1.0)
                    old_group_error = (
                        (
                            fold_group_label_counts[fold][label]
                            - group_target
                        )
                        / group_denominator
                    ) ** 2
                    new_group_error = (
                        (
                            fold_group_label_counts[fold][label]
                            + 1
                            - group_target
                        )
                        / group_denominator
                    ) ** 2
                    delta += (
                        OCCURRENCE_LABEL_WEIGHT * (new_error - old_error)
                        + CONTEXT_GROUP_LABEL_WEIGHT
                        * (new_group_error - old_group_error)
                    ) / label_count

                if delta < best_delta - 1e-12:
                    best_delta = delta
                    best_fold = fold

            assignment[group.group_id] = best_fold
            fold_sizes[best_fold] += group.size
            fold_label_counts[best_fold].update(group.label_counts)
            fold_group_label_counts[best_fold].update(group.label_counts.keys())

        objective = _assignment_objective(
            fold_sizes,
            fold_label_counts,
            fold_group_label_counts,
            total_size,
            total_label_counts,
            total_group_label_counts,
        )
        if objective < best_objective:
            best_objective = objective
            best_assignment = assignment

    assert best_assignment is not None
    return best_assignment


def build_nested_fold_rows(
    sentence_labels_path: str | Path,
    *,
    seed: int = 42,
    outer_splits: int = 5,
    inner_splits: int = 8,
) -> list[dict[str, str | int]]:
    examples = load_sentence_examples(sentence_labels_path)
    groups = group_sentence_examples(examples)
    group_by_id = {group.group_id: group for group in groups}
    outer_assignment = assign_groups_multilabel(
        groups,
        n_splits=outer_splits,
        seed=seed,
    )

    rows: list[dict[str, str | int]] = []
    for outer_fold in range(outer_splits):
        outer_train_groups = [
            group
            for group in groups
            if outer_assignment[group.group_id] != outer_fold
        ]
        inner_assignment = assign_groups_multilabel(
            outer_train_groups,
            n_splits=inner_splits,
            seed=seed + (outer_fold + 1) * 10_007,
        )
        validation_inner_fold = outer_fold % inner_splits

        for group_id, group in group_by_id.items():
            if outer_assignment[group_id] == outer_fold:
                role = "test"
            elif inner_assignment[group_id] == validation_inner_fold:
                role = "validation"
            else:
                role = "train"
            for example in group.examples:
                rows.append(
                    {
                        "protocol_version": "littleprince_context_cv_v1",
                        "seed": seed,
                        "outer_fold": outer_fold,
                        "role": role,
                        "text_embedding_idx": example.text_embedding_idx,
                        "sentence_group_id": group_id,
                        "normalized_text_sha256": example.normalized_text_hash,
                        "chapter": example.chapter,
                        "story_quartile": example.story_quartile,
                        "token_count": example.token_count,
                    }
                )
    return sorted(
        rows,
        key=lambda row: (int(row["outer_fold"]), int(row["text_embedding_idx"])),
    )


def build_keyword_eligibility_rows(
    fold_rows: list[dict[str, str | int]],
    sentence_labels_path: str | Path,
    lexicon_path: str | Path,
) -> list[dict[str, str | int]]:
    sentence_rows = {
        int(row["text_embedding_idx"]): row
        for row in _read_csv(Path(sentence_labels_path))
    }
    lexicon_rows = _read_csv(Path(lexicon_path))
    output: list[dict[str, str | int]] = []

    rows_by_fold: dict[int, list[dict[str, str | int]]] = defaultdict(list)
    for row in fold_rows:
        rows_by_fold[int(row["outer_fold"])].append(row)

    for outer_fold, rows in sorted(rows_by_fold.items()):
        counts: dict[str, Counter[str]] = defaultdict(Counter)
        context_groups: dict[str, dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        for fold_row in rows:
            sentence = sentence_rows[int(fold_row["text_embedding_idx"])]
            present = [
                value for value in sentence["present_keyword_ids"].split("|") if value
            ]
            for keyword_id in set(present):
                role = str(fold_row["role"])
                counts[keyword_id][role] += 1
                context_groups[keyword_id][role].add(
                    str(fold_row["sentence_group_id"])
                )

        for lexicon in lexicon_rows:
            keyword_id = lexicon["keyword_id"]
            train_df = counts[keyword_id]["train"]
            validation_df = counts[keyword_id]["validation"]
            test_df = counts[keyword_id]["test"]
            train_group_df = len(context_groups[keyword_id]["train"])
            validation_group_df = len(context_groups[keyword_id]["validation"])
            test_group_df = len(context_groups[keyword_id]["test"])
            output.append(
                {
                    "outer_fold": outer_fold,
                    "keyword_id": keyword_id,
                    "surface_form": lexicon["surface_form"],
                    "include_core": lexicon["include_core"],
                    "include_main": lexicon["include_main"],
                    "include_extended": lexicon["include_extended"],
                    "story_local_flag": lexicon["story_local_flag"],
                    "train_df": train_df,
                    "validation_df": validation_df,
                    "test_df": test_df,
                    "train_group_df": train_group_df,
                    "validation_group_df": validation_group_df,
                    "test_group_df": test_group_df,
                    "eligible_1_1_1": str(
                        train_df >= 1 and validation_df >= 1 and test_df >= 1
                    ).lower(),
                    "eligible_10_3_3": str(
                        train_df >= 10
                        and validation_df >= 3
                        and test_df >= 3
                    ).lower(),
                    "eligible_20_5_5": str(
                        train_df >= 20
                        and validation_df >= 5
                        and test_df >= 5
                    ).lower(),
                    "eligible_group_1_1_1": str(
                        train_group_df >= 1
                        and validation_group_df >= 1
                        and test_group_df >= 1
                    ).lower(),
                    "eligible_group_10_3_3": str(
                        train_group_df >= 10
                        and validation_group_df >= 3
                        and test_group_df >= 3
                    ).lower(),
                    "eligible_group_20_5_5": str(
                        train_group_df >= 20
                        and validation_group_df >= 5
                        and test_group_df >= 5
                    ).lower(),
                }
            )
    return output


def write_nested_fold_artifacts(
    *,
    sentence_labels_path: str | Path,
    lexicon_path: str | Path,
    fold_output_path: str | Path,
    eligibility_output_path: str | Path,
    provenance_path: str | Path,
    seed: int = 42,
) -> dict[str, object]:
    labels_file = Path(sentence_labels_path)
    lexicon_file = Path(lexicon_path)
    fold_output = Path(fold_output_path)
    eligibility_output = Path(eligibility_output_path)
    provenance_file = Path(provenance_path)

    fold_rows = build_nested_fold_rows(labels_file, seed=seed)
    eligibility_rows = build_keyword_eligibility_rows(
        fold_rows,
        labels_file,
        lexicon_file,
    )

    fold_output.parent.mkdir(parents=True, exist_ok=True)
    with fold_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FOLD_FIELDS)
        writer.writeheader()
        writer.writerows(fold_rows)
    with eligibility_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ELIGIBILITY_FIELDS)
        writer.writeheader()
        writer.writerows(eligibility_rows)

    role_counts: dict[str, dict[str, int]] = {}
    for outer_fold in range(5):
        counts = Counter(
            str(row["role"])
            for row in fold_rows
            if int(row["outer_fold"]) == outer_fold
        )
        role_counts[str(outer_fold)] = dict(sorted(counts.items()))

    group_ids = {
        int(row["text_embedding_idx"]): str(row["sentence_group_id"])
        for row in fold_rows
        if int(row["outer_fold"]) == 0
    }
    duplicate_groups = Counter(group_ids.values())
    summary: dict[str, object] = {
        "schema_version": "littleprince_context_cv_v1",
        "assignment_algorithm_version": "grouped_multilabel_v2",
        "algorithm": (
            "deterministic greedy grouped multilabel stratification over "
            "sentence DF and normalized-context-group DF; "
            "32 fold-order restarts"
        ),
        "objective_weights": {
            "sentence_occurrence_label": OCCURRENCE_LABEL_WEIGHT,
            "normalized_context_group_label": CONTEXT_GROUP_LABEL_WEIGHT,
            "sentence_count": SIZE_WEIGHT,
        },
        "seed": seed,
        "outer_splits": 5,
        "inner_splits": 8,
        "inner_validation_fold": "outer_fold modulo 8",
        "builder_source_sha256": file_sha256(Path(__file__)),
        "python_version": sys.version.split()[0],
        "source_sentence_labels_sha256": file_sha256(labels_file),
        "source_lexicon_sha256": file_sha256(lexicon_file),
        "fold_output_sha256": file_sha256(fold_output),
        "eligibility_output_sha256": file_sha256(eligibility_output),
        "valid_sentences": len(group_ids),
        "sentence_groups": len(set(group_ids.values())),
        "repeated_sentence_groups": sum(
            count > 1 for count in duplicate_groups.values()
        ),
        "largest_sentence_group": max(duplicate_groups.values()),
        "role_counts": role_counts,
    }
    provenance_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
