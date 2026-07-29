from __future__ import annotations

import pytest
import torch

from eeg_keyword_decoding.ot import (
    TransportPoolingConfig,
    transport_pool_eeg_to_words,
)


def test_transport_pooling_shape_mask_and_column_normalization():
    eeg = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    plan = torch.tensor(
        [[[0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.0, 0.5, 0.0]]]
    )
    word_mask = torch.tensor([[True, True, False]])
    output = transport_pool_eeg_to_words(
        eeg_sequence=eeg,
        plan=plan,
        word_mask=word_mask,
    )
    assert output.sequence.shape == (1, 3, 2)
    torch.testing.assert_close(
        output.sequence[0, :2],
        torch.tensor([[2.0, 3.0], [5.0, 6.0]]),
    )
    assert torch.count_nonzero(output.sequence[0, 2]) == 0
    torch.testing.assert_close(
        output.column_mass,
        torch.tensor([[0.5, 0.5, 0.0]]),
    )


def test_near_one_hot_plan_selects_expected_eeg_vectors():
    eeg = torch.tensor([[[1.0], [2.0], [9.0]]])
    plan = torch.tensor(
        [[[0.5, 0.0], [0.0, 0.0], [0.0, 0.5]]],
        requires_grad=True,
    )
    output = transport_pool_eeg_to_words(
        eeg_sequence=eeg,
        plan=plan,
        word_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    torch.testing.assert_close(
        output.sequence.squeeze(-1),
        torch.tensor([[1.0, 9.0]]),
    )


def test_transport_pooling_backpropagates_to_eeg_and_plan():
    eeg = torch.randn(2, 4, 3, requires_grad=True)
    raw = torch.rand(2, 4, 2, requires_grad=True)
    plan = raw / raw.sum(dim=(1, 2), keepdim=True)
    output = transport_pool_eeg_to_words(
        eeg_sequence=eeg,
        plan=plan,
        word_mask=torch.ones(2, 2, dtype=torch.bool),
    )
    output.sequence.square().mean().backward()
    assert eeg.grad is not None and torch.isfinite(eeg.grad).all()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()


def test_transport_pooling_padding_does_not_change_valid_words():
    eeg = torch.randn(1, 3, 4)
    plan = torch.rand(1, 3, 2)
    plan = plan / plan.sum(dim=(1, 2), keepdim=True)
    baseline = transport_pool_eeg_to_words(
        eeg_sequence=eeg,
        plan=plan,
        word_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    padded_eeg = torch.randn(1, 7, 4) * 100_000
    padded_eeg[:, :3] = eeg
    padded_plan = torch.zeros(1, 7, 5)
    padded_plan[:, :3, :2] = plan
    observed = transport_pool_eeg_to_words(
        eeg_sequence=padded_eeg,
        plan=padded_plan,
        word_mask=torch.tensor([[1, 1, 0, 0, 0]], dtype=torch.bool),
    )
    torch.testing.assert_close(
        observed.sequence[:, :2],
        baseline.sequence,
        rtol=0,
        atol=0,
    )
    assert torch.count_nonzero(observed.sequence[:, 2:]) == 0


def test_transport_pooling_rejects_zero_mass_for_valid_word():
    with pytest.raises(ValueError, match="positive mass"):
        transport_pool_eeg_to_words(
            eeg_sequence=torch.randn(1, 2, 3),
            plan=torch.zeros(1, 2, 1),
            word_mask=torch.ones(1, 1, dtype=torch.bool),
        )


def test_transport_pooling_config_contract():
    config = TransportPoolingConfig.from_dict(
        {
            "schema_version": "transport_pooling_v1",
            "normalize_by_column_mass": True,
            "eps": 1e-8,
        }
    )
    assert config.normalize_by_column_mass
    with pytest.raises(ValueError, match="requires"):
        TransportPoolingConfig.from_dict(
            {
                "schema_version": "transport_pooling_v1",
                "normalize_by_column_mass": False,
                "eps": 1e-8,
            }
        )
