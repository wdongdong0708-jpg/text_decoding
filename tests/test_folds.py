import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import eeg_keyword_decoding.data.folds as fold_module
from eeg_keyword_decoding.data import (
    assign_groups_multilabel,
    group_sentence_examples,
    load_sentence_examples,
)
from eeg_keyword_decoding.data.protocol_assets import file_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ROOT = PROJECT_ROOT / "data" / "protocols" / "littleprince_hf_v1"


def test_group_assignment_is_deterministic_and_keeps_duplicate_text_atomic():
    examples = load_sentence_examples(
        PROTOCOL_ROOT / "littleprince_sentence_keyword_labels_v1.csv"
    )
    groups = group_sentence_examples(examples)
    first = assign_groups_multilabel(groups, n_splits=5, seed=42)
    second = assign_groups_multilabel(groups, n_splits=5, seed=42)

    assert first == second
    assert set(first) == {group.group_id for group in groups}
    assert set(first.values()) == set(range(5))
    assert all(len(group_id.removeprefix("lptext:")) == 64 for group_id in first)


def test_generated_nested_folds_are_group_disjoint_and_cover_each_test_once():
    fold_path = PROTOCOL_ROOT / "littleprince_sentence_folds_v1.csv"
    with fold_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2809 * 5
    by_fold: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_fold[int(row["outer_fold"])].append(row)

    test_counts: Counter[int] = Counter()
    for outer_fold, fold_rows in by_fold.items():
        assert {row["role"] for row in fold_rows} == {
            "train",
            "validation",
            "test",
        }
        role_by_group: dict[str, set[str]] = defaultdict(set)
        for row in fold_rows:
            role_by_group[row["sentence_group_id"]].add(row["role"])
            if row["role"] == "test":
                test_counts[int(row["text_embedding_idx"])] += 1
        assert all(len(roles) == 1 for roles in role_by_group.values())
        counts = Counter(row["role"] for row in fold_rows)
        assert 520 <= counts["test"] <= 610
        assert 240 <= counts["validation"] <= 320
        assert sum(counts.values()) == 2809
        assert outer_fold in range(5)

    assert len(test_counts) == 2809
    assert set(test_counts.values()) == {1}


def test_keyword_eligibility_includes_sentence_and_context_group_df():
    eligibility_path = (
        PROTOCOL_ROOT / "littleprince_keyword_fold_eligibility_v1.csv"
    )
    with eligibility_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 247 * 5
    for row in rows:
        for role in ("train", "validation", "test"):
            sentence_df = int(row[f"{role}_df"])
            group_df = int(row[f"{role}_group_df"])
            assert 0 <= group_df <= sentence_df

    core_rows = [row for row in rows if row["include_core"] == "true"]
    assert len(core_rows) == 33 * 5
    assert all(row["eligible_20_5_5"] == "true" for row in core_rows)
    assert all(row["eligible_group_20_5_5"] == "true" for row in core_rows)

    main_rows = [row for row in rows if row["include_main"] == "true"]
    assert len(main_rows) == 64 * 5
    assert min(int(row["test_group_df"]) for row in main_rows) >= 4
    assert min(int(row["validation_group_df"]) for row in main_rows) >= 2


def test_frozen_fold_artifacts_match_recorded_source_and_hashes():
    provenance = json.loads(
        (PROTOCOL_ROOT / "FOLDS_PROVENANCE.json").read_text(encoding="utf-8")
    )
    fold_path = PROTOCOL_ROOT / "littleprince_sentence_folds_v1.csv"
    eligibility_path = (
        PROTOCOL_ROOT / "littleprince_keyword_fold_eligibility_v1.csv"
    )

    assert provenance["assignment_algorithm_version"] == (
        "grouped_multilabel_v2"
    )
    assert provenance["builder_source_sha256"] == file_sha256(
        Path(fold_module.__file__)
    )
    assert provenance["fold_output_sha256"] == file_sha256(fold_path)
    assert provenance["eligibility_output_sha256"] == file_sha256(
        eligibility_path
    )
