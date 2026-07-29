from __future__ import annotations

import pytest
import torch

from eeg_keyword_decoding.models import EEGSequenceEncoder

from test_eeg_sequence_encoder import prefix_mask, small_config


def test_padding_positions_are_strictly_zero():
    model = EEGSequenceEncoder(small_config(output_dim=8)).eval()
    output = model(
        eeg=torch.randn(2, 4, 101),
        eeg_mask=prefix_mask([31, 101], 101),
        subject_indices=torch.tensor([0, 1], dtype=torch.int64),
    )
    assert torch.count_nonzero(output.sequence[~output.mask]) == 0


def test_single_sample_and_mixed_length_batch_have_identical_valid_output():
    torch.manual_seed(7)
    model = EEGSequenceEncoder(small_config(output_dim=8)).eval()
    short = torch.randn(1, 4, 33)
    single = model(
        eeg=short,
        eeg_mask=torch.ones(1, 33, dtype=torch.bool),
        subject_indices=torch.tensor([2], dtype=torch.int64),
    )
    batch_eeg = torch.zeros(2, 4, 101)
    batch_eeg[0, :, :33] = short[0]
    batch_eeg[1] = torch.randn(4, 101)
    mixed = model(
        eeg=batch_eeg,
        eeg_mask=prefix_mask([33, 101], 101),
        subject_indices=torch.tensor([2, 3], dtype=torch.int64),
    )
    valid = int(single.lengths[0])
    torch.testing.assert_close(
        mixed.sequence[0, :valid],
        single.sequence[0, :valid],
        rtol=1e-5,
        atol=1e-6,
    )


def test_extra_zero_right_padding_does_not_change_valid_output():
    torch.manual_seed(11)
    model = EEGSequenceEncoder(small_config(output_dim=8)).eval()
    original = torch.randn(1, 4, 31)
    baseline = model(
        eeg=original,
        eeg_mask=torch.ones(1, 31, dtype=torch.bool),
        subject_indices=torch.tensor([0], dtype=torch.int64),
    )
    padded = torch.zeros(1, 4, 100)
    padded[:, :, :31] = original
    observed = model(
        eeg=padded,
        eeg_mask=prefix_mask([31], 100),
        subject_indices=torch.tensor([0], dtype=torch.int64),
    )
    valid = int(baseline.lengths[0])
    torch.testing.assert_close(
        observed.sequence[:, :valid],
        baseline.sequence[:, :valid],
        rtol=1e-5,
        atol=1e-6,
    )


def test_invalid_padding_values_are_cleared_before_first_convolution():
    torch.manual_seed(13)
    model = EEGSequenceEncoder(small_config(output_dim=8)).eval()
    mask = prefix_mask([31, 101], 101)
    clean = torch.randn(2, 4, 101)
    clean[0, :, 31:] = 0.0
    contaminated = clean.clone()
    contaminated[0, :, 31:] = torch.randn(4, 70) * 100_000
    clean_output = model(
        eeg=clean,
        eeg_mask=mask,
        subject_indices=torch.tensor([0, 1], dtype=torch.int64),
    )
    contaminated_output = model(
        eeg=contaminated,
        eeg_mask=mask,
        subject_indices=torch.tensor([0, 1], dtype=torch.int64),
    )
    torch.testing.assert_close(
        contaminated_output.sequence,
        clean_output.sequence,
        rtol=0,
        atol=0,
    )


def test_padding_input_positions_receive_zero_gradient_from_valid_loss():
    model = EEGSequenceEncoder(small_config(output_dim=8))
    eeg = torch.randn(2, 4, 101, requires_grad=True)
    mask = prefix_mask([31, 101], 101)
    output = model(
        eeg=eeg,
        eeg_mask=mask,
        subject_indices=torch.tensor([0, 1], dtype=torch.int64),
    )
    loss = output.sequence.square().masked_select(
        output.mask.unsqueeze(-1)
    ).mean()
    loss.backward()
    assert eeg.grad is not None
    assert torch.count_nonzero(eeg.grad[0, :, 31:]) == 0
    assert torch.count_nonzero(eeg.grad[0, :, :31]) > 0


def test_batch_permutation_is_equivariant():
    torch.manual_seed(17)
    model = EEGSequenceEncoder(small_config(output_dim=8)).eval()
    lengths = [31, 33, 101]
    eeg = torch.randn(3, 4, 101)
    mask = prefix_mask(lengths, 101)
    subjects = torch.tensor([0, 2, 5], dtype=torch.int64)
    expected = model(
        eeg=eeg,
        eeg_mask=mask,
        subject_indices=subjects,
    )
    permutation = torch.tensor([2, 0, 1])
    inverse = torch.argsort(permutation)
    observed = model(
        eeg=eeg[permutation],
        eeg_mask=mask[permutation],
        subject_indices=subjects[permutation],
    )
    torch.testing.assert_close(
        observed.sequence[inverse],
        expected.sequence,
        rtol=0,
        atol=0,
    )
    assert torch.equal(observed.mask[inverse], expected.mask)


def test_non_contiguous_mask_is_rejected():
    model = EEGSequenceEncoder(small_config(output_dim=8))
    mask = torch.ones(1, 31, dtype=torch.bool)
    mask[0, 10] = False
    with pytest.raises(ValueError, match="contiguous true prefix"):
        model(
            eeg=torch.randn(1, 4, 31),
            eeg_mask=mask,
            subject_indices=torch.tensor([0], dtype=torch.int64),
        )


def test_empty_and_too_short_sequences_are_rejected():
    model = EEGSequenceEncoder(small_config(output_dim=8))
    with pytest.raises(ValueError, match="Empty EEG"):
        model(
            eeg=torch.randn(1, 4, 31),
            eeg_mask=torch.zeros(1, 31, dtype=torch.bool),
            subject_indices=torch.tensor([0], dtype=torch.int64),
        )
    with pytest.raises(ValueError, match="shorter"):
        model(
            eeg=torch.randn(1, 4, 6),
            eeg_mask=torch.ones(1, 6, dtype=torch.bool),
            subject_indices=torch.tensor([0], dtype=torch.int64),
        )


def test_mask_dtype_and_input_channel_errors_are_explicit():
    model = EEGSequenceEncoder(small_config(output_dim=8))
    with pytest.raises(ValueError, match="torch.bool"):
        model(
            eeg=torch.randn(1, 4, 31),
            eeg_mask=torch.ones(1, 31),
            subject_indices=torch.tensor([0], dtype=torch.int64),
        )
    with pytest.raises(ValueError, match="input channels"):
        model(
            eeg=torch.randn(1, 5, 31),
            eeg_mask=torch.ones(1, 31, dtype=torch.bool),
            subject_indices=torch.tensor([0], dtype=torch.int64),
        )
