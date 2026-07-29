import csv
from collections import defaultdict
from pathlib import Path

from eeg_keyword_decoding.data import (
    group_records_by_sentence,
    load_eeg_manifest,
    validate_eeg_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "littleprince_pl_all_clean_manifest.csv"
)


def test_manifest_uses_sentence_occurrence_identity_without_old_target_ids():
    records = load_eeg_manifest(MANIFEST)
    validate_eeg_manifest(records, check_files=False)
    grouped = group_records_by_sentence(records)

    assert len(records) == 21110
    assert sorted(grouped) == list(range(16, 2853))
    assert records[0].sentence_occurrence_id == "littleprince:16"
    assert 6 <= len(grouped[16]) <= 8


def test_all_eeg_views_inherit_their_sentence_fold_role():
    records = load_eeg_manifest(MANIFEST)
    grouped = group_records_by_sentence(records)
    fold_path = (
        PROJECT_ROOT
        / "data"
        / "protocols"
        / "littleprince_hf_v1"
        / "littleprince_sentence_folds_v1.csv"
    )
    with fold_path.open("r", encoding="utf-8", newline="") as handle:
        fold_rows = list(csv.DictReader(handle))

    roles: dict[tuple[int, int], set[str]] = defaultdict(set)
    for row in fold_rows:
        outer_fold = int(row["outer_fold"])
        idx = int(row["text_embedding_idx"])
        assert idx in grouped
        for _record in grouped[idx]:
            roles[(outer_fold, idx)].add(row["role"])

    assert len(roles) == 2809 * 5
    assert all(len(values) == 1 for values in roles.values())
    assert min(len(grouped[idx]) for _, idx in roles) == 6
    assert max(len(grouped[idx]) for _, idx in roles) == 8
