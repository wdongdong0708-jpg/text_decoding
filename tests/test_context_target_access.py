from __future__ import annotations

import pytest

from eeg_keyword_decoding.data import (
    ContextEEGDataset,
    ContextTargetAccessError,
)

from conftest import PROTOCOL_ROOT


def _arguments(split_index, subject_index, keyword_index):
    return {
        "split_index": split_index,
        "outer_fold": 0,
        "subject_index": subject_index,
        "keyword_index": keyword_index,
        "sentence_labels_path": (
            PROTOCOL_ROOT / "littleprince_sentence_keyword_labels_v1.csv"
        ),
        "word_occurrences_path": (
            PROTOCOL_ROOT / "littleprince_word_occurrences_v1.csv"
        ),
        "text_backend": "bge_m3",
    }


@pytest.mark.parametrize("role", ["validation", "test"])
def test_validation_and_test_cannot_request_context_targets(
    split_index,
    subject_index,
    keyword_index,
    role,
):
    with pytest.raises(ContextTargetAccessError, match="inner-train"):
        ContextEEGDataset(
            **_arguments(split_index, subject_index, keyword_index),
            role=role,
            include_context_targets=True,
        )


@pytest.mark.parametrize("role", ["validation", "test"])
def test_validation_and_test_cannot_be_wired_to_context_store(
    split_index,
    subject_index,
    keyword_index,
    synthetic_context_stores,
    role,
):
    with pytest.raises(ContextTargetAccessError, match="must not be wired"):
        ContextEEGDataset(
            **_arguments(split_index, subject_index, keyword_index),
            role=role,
            include_context_targets=False,
            context_store_root=synthetic_context_stores["bge_m3"],
        )


def test_train_context_targets_require_explicit_store(
    split_index,
    subject_index,
    keyword_index,
):
    with pytest.raises(ValueError, match="context_store_root"):
        ContextEEGDataset(
            **_arguments(split_index, subject_index, keyword_index),
            role="train",
            include_context_targets=True,
        )
