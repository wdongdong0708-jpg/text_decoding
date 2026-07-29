from __future__ import annotations

import pytest
import torch

from eeg_keyword_decoding.models import EEGSequenceEncoder, SubjectFiLM

from test_eeg_sequence_encoder import prefix_mask, small_config


def test_film_gamma_and_beta_initialization():
    adapter = SubjectFiLM(num_subjects=8, feature_dim=16)
    torch.testing.assert_close(
        adapter.gamma.weight,
        torch.ones_like(adapter.gamma.weight),
    )
    torch.testing.assert_close(
        adapter.beta.weight,
        torch.zeros_like(adapter.beta.weight),
    )


def test_enabled_subject_adapter_can_calibrate_subjects_differently():
    model = EEGSequenceEncoder(small_config(output_dim=8)).eval()
    assert model.subject_adapter is not None
    with torch.no_grad():
        model.subject_adapter.gamma.weight[1].fill_(1.2)
        model.subject_adapter.beta.weight[1].fill_(0.3)
    eeg = torch.randn(1, 4, 31).repeat(2, 1, 1)
    output = model(
        eeg=eeg,
        eeg_mask=torch.ones(2, 31, dtype=torch.bool),
        subject_indices=torch.tensor([0, 1], dtype=torch.int64),
    )
    assert not torch.allclose(output.sequence[0], output.sequence[1])


def test_disabled_subject_adapter_ignores_subject_identity():
    model = EEGSequenceEncoder(
        small_config(output_dim=8, subject_enabled=False)
    ).eval()
    eeg = torch.randn(1, 4, 31).repeat(2, 1, 1)
    output = model(
        eeg=eeg,
        eeg_mask=torch.ones(2, 31, dtype=torch.bool),
        subject_indices=torch.tensor([0, 999], dtype=torch.int64),
    )
    torch.testing.assert_close(
        output.sequence[0],
        output.sequence[1],
        rtol=0,
        atol=0,
    )


def test_enabled_subject_adapter_requires_valid_global_index():
    model = EEGSequenceEncoder(small_config(output_dim=8))
    with pytest.raises(IndexError, match="out of range"):
        model(
            eeg=torch.randn(1, 4, 31),
            eeg_mask=prefix_mask([31], 31),
            subject_indices=torch.tensor([8], dtype=torch.int64),
        )
    with pytest.raises(ValueError, match="required"):
        model(
            eeg=torch.randn(1, 4, 31),
            eeg_mask=prefix_mask([31], 31),
            subject_indices=None,
        )
