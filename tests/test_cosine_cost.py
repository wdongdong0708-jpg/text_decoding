from __future__ import annotations

import pytest
import torch

from eeg_keyword_decoding.ot import masked_cosine_cost


def test_cosine_cost_known_unit_vector_relationships():
    eeg = torch.tensor([[[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]])
    text = torch.tensor([[[1.0, 0.0]]])
    output = masked_cosine_cost(
        eeg_sequence=eeg,
        text_sequence=text,
        eeg_mask=torch.ones(1, 3, dtype=torch.bool),
        word_mask=torch.ones(1, 1, dtype=torch.bool),
    )
    torch.testing.assert_close(
        output.cost[0, :, 0],
        torch.tensor([0.0, 2.0, 1.0]),
    )


def test_cosine_cost_masks_invalid_pairs_with_finite_nonzero_sentinel():
    output = masked_cosine_cost(
        eeg_sequence=torch.randn(1, 3, 4),
        text_sequence=torch.randn(1, 3, 4),
        eeg_mask=torch.tensor([[True, True, False]]),
        word_mask=torch.tensor([[True, False, False]]),
    )
    assert output.valid_pair_mask.sum() == 2
    assert torch.isfinite(output.cost).all()
    assert torch.all(output.cost[~output.valid_pair_mask] == 2.0)


def test_cosine_cost_rejects_dimension_and_nonfinite_inputs():
    with pytest.raises(ValueError, match="dimensions differ"):
        masked_cosine_cost(
            eeg_sequence=torch.randn(1, 2, 4),
            text_sequence=torch.randn(1, 2, 5),
            eeg_mask=torch.ones(1, 2, dtype=torch.bool),
            word_mask=torch.ones(1, 2, dtype=torch.bool),
        )
    eeg = torch.randn(1, 2, 4)
    eeg[0, 1, 0] = torch.inf
    with pytest.raises(ValueError, match="NaN or Inf"):
        masked_cosine_cost(
            eeg_sequence=eeg,
            text_sequence=torch.randn(1, 2, 4),
            eeg_mask=torch.ones(1, 2, dtype=torch.bool),
            word_mask=torch.ones(1, 2, dtype=torch.bool),
        )


def test_cosine_cost_extra_padding_does_not_change_valid_region():
    torch.manual_seed(8)
    eeg = torch.randn(1, 3, 5)
    text = torch.randn(1, 2, 5)
    baseline = masked_cosine_cost(
        eeg_sequence=eeg,
        text_sequence=text,
        eeg_mask=torch.ones(1, 3, dtype=torch.bool),
        word_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    padded_eeg = torch.randn(1, 7, 5) * 10_000
    padded_text = torch.randn(1, 6, 5) * 10_000
    padded_eeg[:, :3] = eeg
    padded_text[:, :2] = text
    observed = masked_cosine_cost(
        eeg_sequence=padded_eeg,
        text_sequence=padded_text,
        eeg_mask=torch.tensor([[1, 1, 1, 0, 0, 0, 0]], dtype=torch.bool),
        word_mask=torch.tensor([[1, 1, 0, 0, 0, 0]], dtype=torch.bool),
    )
    torch.testing.assert_close(
        observed.cost[:, :3, :2],
        baseline.cost,
        rtol=0,
        atol=0,
    )
