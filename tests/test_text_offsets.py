import numpy as np
import pytest

from eeg_keyword_decoding.text import (
    ContextWordOccurrence,
    OffsetAlignmentError,
    aggregate_token_features,
    align_word_to_tokens,
)


def _occurrence(
    *,
    char_start: int = 0,
    char_end: int = 3,
) -> ContextWordOccurrence:
    return ContextWordOccurrence(
        word_occurrence_id="lp:1:word:0",
        text_embedding_idx=1,
        word_position=0,
        surface_form="甲乙丙",
        char_start=char_start,
        char_end=char_end,
        keyword_id="",
    )


def test_overlap_weighted_aggregation_excludes_special_and_padding_tokens():
    alignment = align_word_to_tokens(
        _occurrence(),
        offset_mapping=[(0, 0), (0, 2), (2, 3), (0, 0), (0, 0)],
        special_tokens_mask=[1, 0, 0, 1, 0],
        attention_mask=[1, 1, 1, 1, 0],
    )
    assert alignment.token_indices == (1, 2)
    assert alignment.overlap_lengths == (2, 1)

    features = np.asarray(
        [
            [[100.0, 100.0]],
            [[1.0, 4.0]],
            [[7.0, 10.0]],
            [[200.0, 200.0]],
            [[300.0, 300.0]],
        ],
        dtype=np.float32,
    )
    aggregated = aggregate_token_features(features, [alignment])
    np.testing.assert_allclose(
        aggregated,
        np.asarray([[[3.0, 6.0]]], dtype=np.float32),
    )


def test_alignment_fails_when_word_characters_are_not_fully_covered():
    with pytest.raises(OffsetAlignmentError, match="missing character"):
        align_word_to_tokens(
            _occurrence(),
            offset_mapping=[(0, 0), (0, 1), (2, 3), (0, 0)],
            special_tokens_mask=[1, 0, 0, 1],
        )


def test_alignment_fails_if_special_token_overlaps_a_word():
    with pytest.raises(OffsetAlignmentError, match="Special token"):
        align_word_to_tokens(
            _occurrence(char_start=0, char_end=1),
            offset_mapping=[(0, 1)],
            special_tokens_mask=[1],
        )
