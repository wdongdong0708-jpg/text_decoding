from __future__ import annotations

import copy
import io
from pathlib import Path

import pytest
import torch
from torch import nn

from eeg_keyword_decoding.models import (
    EEGSequenceEncoder,
    EEGSequenceEncoderConfig,
    build_eeg_sequence_encoder,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs" / "models" / "eeg_sequence_conv_v1.yaml"
)


def small_config(
    *,
    input_channels: int = 4,
    hidden_dim: int = 16,
    output_dim: int = 32,
    subject_enabled: bool = True,
    dropout: float = 0.0,
) -> EEGSequenceEncoderConfig:
    return EEGSequenceEncoderConfig.from_dict(
        {
            "schema_version": "eeg_sequence_conv_v1",
            "input_channels": input_channels,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "total_stride": 4,
            "minimum_input_length": 7,
            "stem": {"kernel_size": 7, "stride": 2},
            "residual_blocks": {
                "kernel_size": 5,
                "dilations": [1, 2, 4, 8],
                "count": 4,
            },
            "downsample": {"kernel_size": 5, "stride": 2},
            "dropout": dropout,
            "normalization": "timewise_layer_norm",
            "activation": "gelu",
            "convolution_mode": "symmetric_noncausal",
            "parameter_initialization": "pytorch_default",
            "mask_rule": "zero_after_every_operation",
            "subject_adapter": {
                "enabled": subject_enabled,
                "type": "film",
                "num_subjects": 8,
                "gamma_initialization": 1.0,
                "beta_initialization": 0.0,
            },
        }
    )


def prefix_mask(lengths: list[int], maximum: int) -> torch.Tensor:
    return torch.arange(maximum).unsqueeze(0) < torch.tensor(
        lengths,
        dtype=torch.int64,
    ).unsqueeze(1)


@pytest.mark.parametrize(
    ("batch_size", "input_channels", "maximum_length"),
    [(1, 2, 31), (3, 4, 101), (2, 7, 1297)],
)
def test_encoder_preserves_sequence_axis_and_returns_256_features(
    batch_size: int,
    input_channels: int,
    maximum_length: int,
):
    config = small_config(
        input_channels=input_channels,
        hidden_dim=16,
        output_dim=256,
    )
    model = EEGSequenceEncoder(config).eval()
    lengths = [maximum_length - index for index in range(batch_size)]
    eeg = torch.randn(batch_size, input_channels, maximum_length)
    output = model(
        eeg=eeg,
        eeg_mask=prefix_mask(lengths, maximum_length),
        subject_indices=torch.arange(batch_size, dtype=torch.int64),
    )

    assert output.sequence.shape == (
        batch_size,
        (maximum_length + 3) // 4,
        256,
    )
    assert output.mask.shape == output.sequence.shape[:2]
    assert output.mask.dtype == torch.bool
    assert output.lengths.dtype == torch.int64
    assert output.lengths.tolist() == [
        (length + 3) // 4 for length in lengths
    ]
    assert output.mask.sum(dim=1).tolist() == output.lengths.tolist()


def test_stride_four_length_formula_covers_odd_even_and_real_extremes():
    lengths = [7, 31, 32, 33, 100, 101, 1297]
    maximum = max(lengths)
    model = EEGSequenceEncoder(
        small_config(input_channels=3, output_dim=8)
    ).eval()
    output = model(
        eeg=torch.randn(len(lengths), 3, maximum),
        eeg_mask=prefix_mask(lengths, maximum),
        subject_indices=torch.arange(len(lengths), dtype=torch.int64),
    )
    assert output.lengths.tolist() == [
        (length + 3) // 4 for length in lengths
    ]
    assert output.sequence.shape[1] == 325


def test_model_uses_only_timewise_layer_norm():
    model = EEGSequenceEncoder(small_config())
    assert any(isinstance(module, nn.LayerNorm) for module in model.modules())
    assert not any(
        isinstance(module, (nn.BatchNorm1d, nn.GroupNorm))
        for module in model.modules()
    )


def test_virtual_masked_loss_backpropagates_finite_gradients():
    model = EEGSequenceEncoder(small_config())
    mask = prefix_mask([31, 29], 31)
    output = model(
        eeg=torch.randn(2, 4, 31),
        eeg_mask=mask,
        subject_indices=torch.tensor([0, 1], dtype=torch.int64),
    )
    valid = output.mask.unsqueeze(-1)
    loss = output.sequence.square().masked_select(valid).mean()
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert model.channel_projection.weight.grad is not None


def test_state_dict_round_trip_preserves_eval_output():
    config = small_config(dropout=0.1)
    first = EEGSequenceEncoder(config).eval()
    eeg = torch.randn(2, 4, 33)
    mask = prefix_mask([31, 33], 33)
    subjects = torch.tensor([1, 2], dtype=torch.int64)
    expected = first(
        eeg=eeg,
        eeg_mask=mask,
        subject_indices=subjects,
    )

    buffer = io.BytesIO()
    torch.save(first.state_dict(), buffer)
    buffer.seek(0)
    second = EEGSequenceEncoder(copy.deepcopy(config)).eval()
    second.load_state_dict(torch.load(buffer, weights_only=True))
    observed = second(
        eeg=eeg,
        eeg_mask=mask,
        subject_indices=subjects,
    )
    torch.testing.assert_close(observed.sequence, expected.sequence)
    assert torch.equal(observed.mask, expected.mask)
    assert torch.equal(observed.lengths, expected.lengths)


def test_yaml_builder_validates_audited_data_contract():
    model = build_eeg_sequence_encoder(
        DEFAULT_CONFIG,
        actual_input_channels=128,
        actual_num_subjects=8,
    )
    assert model.config.input_channels == 128
    assert model.config.subject_adapter.num_subjects == 8
    assert model.to_config()["total_stride"] == 4
    assert len(model.config.canonical_sha256) == 64
    with pytest.raises(ValueError, match="Dataset channels"):
        build_eeg_sequence_encoder(
            DEFAULT_CONFIG,
            actual_input_channels=127,
            actual_num_subjects=8,
        )
    with pytest.raises(ValueError, match="Dataset subjects"):
        build_eeg_sequence_encoder(
            DEFAULT_CONFIG,
            actual_input_channels=128,
            actual_num_subjects=7,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_autocast_forward_and_backward_are_finite():
    device = torch.device("cuda")
    model = EEGSequenceEncoder(
        small_config(input_channels=8, hidden_dim=32, output_dim=64)
    ).to(device)
    eeg = torch.randn(2, 8, 1297, device=device)
    mask = prefix_mask([1297, 1001], 1297).to(device)
    subjects = torch.tensor([0, 7], dtype=torch.int64, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model(
            eeg=eeg,
            eeg_mask=mask,
            subject_indices=subjects,
        )
        loss = output.sequence.float().square().mean()
    loss.backward()
    assert torch.isfinite(output.sequence).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
