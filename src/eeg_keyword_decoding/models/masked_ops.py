from __future__ import annotations

import torch


def lengths_from_prefix_mask(mask: torch.Tensor) -> torch.Tensor:
    """Validate a right-padded boolean mask and return int64 lengths."""

    if mask.ndim != 2:
        raise ValueError(
            f"mask must have shape [batch, time], got {tuple(mask.shape)}"
        )
    if mask.dtype != torch.bool:
        raise ValueError(f"mask must be torch.bool, got {mask.dtype}")
    if mask.shape[0] <= 0 or mask.shape[1] <= 0:
        raise ValueError(
            f"mask must have non-empty batch/time axes, got {tuple(mask.shape)}"
        )
    lengths = mask.sum(dim=1, dtype=torch.int64)
    if bool((lengths == 0).any()):
        empty_rows = torch.nonzero(lengths == 0, as_tuple=False).flatten()
        raise ValueError(
            "Empty EEG sequences are forbidden; all-false mask rows: "
            f"{empty_rows.tolist()}"
        )
    expected = lengths_to_mask(lengths, maximum_length=mask.shape[1])
    if not torch.equal(mask, expected):
        invalid_rows = torch.nonzero(
            (mask != expected).any(dim=1),
            as_tuple=False,
        ).flatten()
        raise ValueError(
            "eeg_mask must be a contiguous true prefix followed by false "
            f"padding; invalid rows: {invalid_rows.tolist()}"
        )
    return lengths


def lengths_to_mask(
    lengths: torch.Tensor,
    *,
    maximum_length: int,
) -> torch.Tensor:
    if lengths.ndim != 1:
        raise ValueError(
            f"lengths must have shape [batch], got {tuple(lengths.shape)}"
        )
    if lengths.dtype != torch.int64:
        raise ValueError(f"lengths must be torch.int64, got {lengths.dtype}")
    if maximum_length <= 0:
        raise ValueError("maximum_length must be positive")
    if bool((lengths < 0).any()) or bool((lengths > maximum_length).any()):
        raise ValueError(
            "lengths must lie in [0, maximum_length], got "
            f"{lengths.tolist()} with maximum_length={maximum_length}"
        )
    positions = torch.arange(maximum_length, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


def conv1d_output_lengths(
    lengths: torch.Tensor,
    *,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int = 1,
) -> torch.Tensor:
    if lengths.ndim != 1 or lengths.dtype != torch.int64:
        raise ValueError("lengths must be a one-dimensional int64 tensor")
    if kernel_size <= 0 or stride <= 0 or dilation <= 0 or padding < 0:
        raise ValueError("Invalid Conv1d length parameters")
    numerator = (
        lengths
        + 2 * padding
        - dilation * (kernel_size - 1)
        - 1
    )
    output = torch.div(numerator, stride, rounding_mode="floor") + 1
    if bool((output <= 0).any()):
        raise ValueError(
            "Conv1d configuration produces a non-positive output length: "
            f"{output.tolist()}"
        )
    return output


def mask_channel_first(
    sequence: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if sequence.ndim != 3:
        raise ValueError(
            "channel-first sequence must have shape [batch, channel, time]"
        )
    if mask.shape != (sequence.shape[0], sequence.shape[2]):
        raise ValueError(
            "mask shape does not match channel-first sequence: "
            f"{tuple(mask.shape)} vs {tuple(sequence.shape)}"
        )
    return sequence.masked_fill(~mask.unsqueeze(1), 0.0)


def mask_time_first(
    sequence: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if sequence.ndim != 3:
        raise ValueError(
            "time-first sequence must have shape [batch, time, feature]"
        )
    if mask.shape != sequence.shape[:2]:
        raise ValueError(
            "mask shape does not match time-first sequence: "
            f"{tuple(mask.shape)} vs {tuple(sequence.shape)}"
        )
    return sequence.masked_fill(~mask.unsqueeze(-1), 0.0)
