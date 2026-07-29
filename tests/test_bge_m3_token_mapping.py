import pytest

from eeg_keyword_decoding.text import colbert_source_token_indices


def test_colbert_rows_map_to_tokens_after_cls_and_retain_sep():
    indices = colbert_source_token_indices(
        attention_mask=[1, 1, 1, 1, 0, 0],
        special_tokens_mask=[1, 0, 0, 1, 1, 1],
        colbert_row_count=3,
    )
    assert indices == (1, 2, 3)


def test_colbert_mapping_rejects_unproven_row_count():
    with pytest.raises(ValueError, match="row count"):
        colbert_source_token_indices(
            attention_mask=[1, 1, 1, 1],
            special_tokens_mask=[1, 0, 0, 1],
            colbert_row_count=2,
        )


def test_colbert_mapping_requires_first_attended_token_to_be_special():
    with pytest.raises(ValueError, match="special CLS"):
        colbert_source_token_indices(
            attention_mask=[1, 1],
            special_tokens_mask=[0, 1],
            colbert_row_count=1,
        )
