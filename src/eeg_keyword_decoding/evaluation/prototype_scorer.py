from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from eeg_keyword_decoding.prototypes.schema import (
    MASTER_KEYWORD_COUNT,
    PrototypeBank,
)

UNAVAILABLE_SCORE = -1.0e9


@dataclass(frozen=True)
class PrototypeScoreOutput:
    """Fixed-width keyword scores and the train-only availability mask."""

    scores: torch.Tensor
    available_mask: torch.Tensor


class TextFreePrototypeScorer(nn.Module):
    """Score EEG states against train-only prototypes without textual inputs."""

    def __init__(self, *, temperature: float = 0.07, eps: float = 1e-8) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.temperature = float(temperature)
        self.eps = float(eps)

    def forward(
        self,
        eeg_sequence: torch.Tensor,
        eeg_mask: torch.Tensor,
        prototype_bank: PrototypeBank,
    ) -> PrototypeScoreOutput:
        if eeg_sequence.ndim != 3:
            raise ValueError("eeg_sequence must have shape [B,T,D]")
        if eeg_mask.shape != eeg_sequence.shape[:2]:
            raise ValueError("eeg_mask must have shape [B,T]")
        if eeg_mask.dtype != torch.bool:
            raise ValueError("eeg_mask must be torch.bool")
        if eeg_sequence.shape[-1] != prototype_bank.vectors.shape[-1]:
            raise ValueError("EEG and prototype dimensions differ")
        if prototype_bank.vectors.shape[0] != MASTER_KEYWORD_COUNT:
            raise ValueError("Prototype bank must retain all 247 rows")
        if prototype_bank.vectors.device != eeg_sequence.device:
            raise ValueError("Prototype bank and EEG sequence must share a device")
        if not bool(eeg_mask.any(dim=1).all()):
            raise ValueError("Every sample must contain an EEG time point")

        # Prefix masks make padding semantics explicit and catch accidental holes.
        transitions = eeg_mask[:, 1:].to(torch.int8) - eeg_mask[:, :-1].to(torch.int8)
        if bool((transitions > 0).any()):
            raise ValueError("eeg_mask must be a valid-prefix mask")

        # Keep cosine and reduction in FP32 even when the encoder is autocast.
        states = F.normalize(eeg_sequence.float(), p=2, dim=-1, eps=self.eps)
        prototypes = F.normalize(
            prototype_bank.vectors.float(), p=2, dim=-1, eps=self.eps
        )
        logits = torch.einsum("btd,vd->btv", states, prototypes)
        scaled = logits / self.temperature
        scaled = scaled.masked_fill(~eeg_mask.unsqueeze(-1), -torch.inf)
        valid_counts = eeg_mask.sum(dim=1, dtype=torch.float32)
        scores = self.temperature * (
            torch.logsumexp(scaled, dim=1) - valid_counts.log().unsqueeze(-1)
        )

        available = prototype_bank.available_mask.to(eeg_sequence.device)
        # A separate mask carries semantics; the finite sentinel must also remain
        # finite under multi-row means during occurrence/group aggregation.
        scores = scores.masked_fill(~available.unsqueeze(0), UNAVAILABLE_SCORE)
        if scores.shape != (eeg_sequence.shape[0], MASTER_KEYWORD_COUNT):
            raise AssertionError("Scorer changed the fixed Master candidate width")
        if not bool(torch.isfinite(scores).all()):
            raise FloatingPointError("Non-finite validation keyword score")
        return PrototypeScoreOutput(scores=scores, available_mask=available)
