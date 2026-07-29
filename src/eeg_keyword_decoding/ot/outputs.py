from __future__ import annotations

from dataclasses import dataclass

import torch

from eeg_keyword_decoding.models import TextProjectionOutput


@dataclass(frozen=True)
class CosineCostOutput:
    cost: torch.Tensor
    valid_pair_mask: torch.Tensor


@dataclass(frozen=True)
class SinkhornOutput:
    plan: torch.Tensor
    transport_cost: torch.Tensor
    entropy: torch.Tensor
    row_marginal: torch.Tensor
    column_marginal: torch.Tensor
    row_error: torch.Tensor
    column_error: torch.Tensor
    iterations: int
    epsilon: float


@dataclass(frozen=True)
class TransportPoolOutput:
    sequence: torch.Tensor
    mask: torch.Tensor
    column_mass: torch.Tensor


@dataclass(frozen=True)
class ContextOTOutput:
    text: TextProjectionOutput
    cost: CosineCostOutput
    sinkhorn: SinkhornOutput
    word_conditioned_eeg: TransportPoolOutput
