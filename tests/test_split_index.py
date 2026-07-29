from __future__ import annotations

from collections import Counter, defaultdict


def test_split_view_index_excludes_invalid_sentences_and_joins_every_view(
    split_index,
):
    assert len(split_index.manifest_records) == 21_110
    assert len(split_index.valid_text_embedding_indices) == 2_809
    assert len(split_index.excluded_text_embedding_indices) == 28
    assert split_index.valid_eeg_view_count == 20_902
    assert split_index.excluded_eeg_view_count == 208
    counts = Counter(
        record.text_embedding_idx
        for record in split_index.valid_manifest_records
    )
    assert min(counts.values()) == 6
    assert max(counts.values()) == 8


def test_split_view_index_role_counts_match_frozen_protocol(split_index):
    assert split_index.role_counts() == {
        0: {
            "train": {"sentences": 1967, "eeg_views": 14650},
            "validation": {"sentences": 281, "eeg_views": 2086},
            "test": {"sentences": 561, "eeg_views": 4166},
        },
        1: {
            "train": {"sentences": 1967, "eeg_views": 14630},
            "validation": {"sentences": 280, "eeg_views": 2090},
            "test": {"sentences": 562, "eeg_views": 4182},
        },
        2: {
            "train": {"sentences": 1966, "eeg_views": 14624},
            "validation": {"sentences": 281, "eeg_views": 2093},
            "test": {"sentences": 562, "eeg_views": 4185},
        },
        3: {
            "train": {"sentences": 1966, "eeg_views": 14641},
            "validation": {"sentences": 281, "eeg_views": 2076},
            "test": {"sentences": 562, "eeg_views": 4185},
        },
        4: {
            "train": {"sentences": 1966, "eeg_views": 14620},
            "validation": {"sentences": 281, "eeg_views": 2098},
            "test": {"sentences": 562, "eeg_views": 4184},
        },
    }


def test_all_sentence_views_inherit_one_role_and_test_exactly_once(
    split_index,
):
    test_counts: Counter[int] = Counter()
    roles_by_fold_sentence: dict[tuple[int, int], set[str]] = defaultdict(set)
    for outer_fold in split_index.outer_folds:
        for role in ("train", "validation", "test"):
            for record in split_index.records_for(outer_fold, role):
                roles_by_fold_sentence[
                    (outer_fold, record.text_embedding_idx)
                ].add(role)
                if role == "test":
                    test_counts[record.text_embedding_idx] = 1
    assert len(roles_by_fold_sentence) == 2_809 * 5
    assert all(len(roles) == 1 for roles in roles_by_fold_sentence.values())
    assert set(test_counts) == set(split_index.valid_text_embedding_indices)


def test_subject_index_is_global_and_deterministic(subject_index):
    assert subject_index.subjects == tuple(
        f"sub-{index:02d}" for index in range(1, 9)
    )
    assert subject_index.index("sub-01") == 0
    assert subject_index.index("sub-08") == 7
    assert subject_index.to_metadata()["ordering_rule"] == (
        "ascending Unicode code-point order"
    )


def test_master_keyword_index_has_one_nested_stable_space(keyword_index):
    assert len(keyword_index.master_indices) == 247
    assert len(keyword_index.core_indices) == 33
    assert len(keyword_index.main_indices) == 64
    assert len(keyword_index.extended_indices) == 100
    assert (
        keyword_index.core_indices
        <= keyword_index.main_indices
        <= keyword_index.extended_indices
        <= keyword_index.master_indices
    )
    assert keyword_index.index_or_minus_one("") == -1
    assert len(keyword_index.mapping_sha256) == 64
