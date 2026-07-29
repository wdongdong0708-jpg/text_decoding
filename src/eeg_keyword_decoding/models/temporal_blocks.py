from __future__ import annotations

import torch
from torch import nn

from .masked_ops import (
    conv1d_output_lengths,
    lengths_to_mask,
    mask_channel_first,
)


def make_activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name!r}")


class TimewiseLayerNorm(nn.Module):
    """LayerNorm over channels independently at every time position."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3:
            raise ValueError(
                "TimewiseLayerNorm expects [batch, feature, time]"
            )
        return self.norm(sequence.transpose(1, 2)).transpose(1, 2)


class MaskedTemporalDownsample(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int,
        kernel_size: int,
        stride: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("downsample kernel_size must be a positive odd")
        if stride <= 1:
            raise ValueError("downsample stride must exceed one")
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = self.kernel_size // 2
        self.norm = TimewiseLayerNorm(feature_dim)
        self.convolution = nn.Conv1d(
            feature_dim,
            feature_dim,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        self.activation = make_activation(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        sequence: torch.Tensor,
        mask: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = mask_channel_first(self.norm(sequence), mask)
        output = self.convolution(normalized)
        output_lengths = conv1d_output_lengths(
            lengths,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        output_mask = lengths_to_mask(
            output_lengths,
            maximum_length=output.shape[-1],
        )
        output = mask_channel_first(output, output_mask)
        output = self.activation(output)
        output = mask_channel_first(output, output_mask)
        output = self.dropout(output)
        output = mask_channel_first(output, output_mask)
        return output, output_mask, output_lengths


class MaskedDilatedResidualBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("residual kernel_size must be a positive odd")
        if dilation <= 0:
            raise ValueError("dilation must be positive")
        padding = dilation * (kernel_size - 1) // 2
        self.norm = TimewiseLayerNorm(hidden_dim)
        self.temporal = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.activation = make_activation(activation)
        self.pointwise = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        sequence: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = mask_channel_first(sequence, mask)
        update = mask_channel_first(self.norm(residual), mask)
        update = self.temporal(update)
        update = mask_channel_first(update, mask)
        update = self.activation(update)
        update = mask_channel_first(update, mask)
        update = self.pointwise(update)
        update = mask_channel_first(update, mask)
        update = self.dropout(update)
        update = mask_channel_first(update, mask)
        return mask_channel_first(residual + update, mask)
