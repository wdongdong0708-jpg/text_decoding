from __future__ import annotations

import torch
from torch.nn import functional as F

from .outputs import CosineCostOutput


def _validate_sequence_and_mask(
    sequence: torch.Tensor,
    mask: torch.Tensor,
    *,
    name: str,
) -> None:
    if sequence.ndim != 3:
        raise ValueError(
            f"{name} must have shape [batch, position, feature], got "
            f"{tuple(sequence.shape)}"
        )
    if not sequence.is_floating_point():
        raise ValueError(f"{name} must be floating point")
    if mask.shape != sequence.shape[:2]:
        raise ValueError(
            f"{name}_mask shape {tuple(mask.shape)} does not match "
            f"{tuple(sequence.shape[:2])}"
        )
    if mask.dtype != torch.bool:
        raise ValueError(f"{name}_mask must be torch.bool")
    if mask.device != sequence.device:
        raise ValueError(f"{name} and its mask must share a device")
    if not bool(torch.isfinite(sequence).all()):
        raise ValueError(f"{name} contains NaN or Inf")
    if bool((mask.sum(dim=1) == 0).any()):
        raise ValueError(f"{name} contains an empty valid sequence")


def masked_cosine_cost(
    *,
    eeg_sequence: torch.Tensor,
    text_sequence: torch.Tensor,
    eeg_mask: torch.Tensor,
    word_mask: torch.Tensor,
    eps: float = 1e-8,
    invalid_cost: float = 2.0,
) -> CosineCostOutput:
    """Return finite cosine costs and an explicit valid-pair mask.

    Invalid entries use the finite maximum cosine distance for diagnostics,
    but the returned pair mask—not that sentinel value—excludes them from OT.
    """

    if eps <= 0:
        raise ValueError("eps must be positive")
    if not 0.0 <= invalid_cost <= 2.0:
        raise ValueError("invalid_cost must lie in the cosine-cost range [0, 2]")
    _validate_sequence_and_mask(eeg_sequence, eeg_mask, name="eeg_sequence")
    _validate_sequence_and_mask(
        text_sequence,
        word_mask,
        name="text_sequence",
    )
    if eeg_sequence.shape[0] != text_sequence.shape[0]:
        raise ValueError("EEG and text batch sizes differ")
    if eeg_sequence.shape[-1] != text_sequence.shape[-1]:
        raise ValueError(
            "EEG and text embedding dimensions differ: "
            f"{eeg_sequence.shape[-1]} != {text_sequence.shape[-1]}"
        )
    if eeg_sequence.device != text_sequence.device:
        raise ValueError("EEG and text sequences must share a device")

    eeg = F.normalize(eeg_sequence.float(), p=2, dim=-1, eps=eps)
    text = F.normalize(text_sequence.float(), p=2, dim=-1, eps=eps)
    cosine = torch.bmm(eeg, text.transpose(1, 2))
    cost = 1.0 - cosine.clamp(min=-1.0, max=1.0)
    pair_mask = eeg_mask.unsqueeze(2) & word_mask.unsqueeze(1)
    cost = cost.masked_fill(~pair_mask, float(invalid_cost))
    if not bool(torch.isfinite(cost).all()):
        raise FloatingPointError("Cosine cost produced NaN or Inf")
    return CosineCostOutput(cost=cost, valid_pair_mask=pair_mask)
