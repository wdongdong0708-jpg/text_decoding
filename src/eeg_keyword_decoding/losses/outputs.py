from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class OTContextLossOutput:
    loss: torch.Tensor
    per_sample_loss: torch.Tensor
    plan_total_mass: torch.Tensor
    entropy: torch.Tensor


@dataclass(frozen=True)
class ContextTokenLossOutput:
    loss: torch.Tensor
    eeg_to_text_loss: torch.Tensor
    text_to_eeg_loss: torch.Tensor
    eeg_to_text_query_loss: torch.Tensor
    text_to_eeg_query_loss: torch.Tensor
    eeg_to_text_sample_loss: torch.Tensor
    text_to_eeg_sample_loss: torch.Tensor
    valid_words_per_sample: torch.Tensor
    positive_counts: torch.Tensor
    negative_counts: torch.Tensor
    valid_query_count: int
    contributing_query_count: int
    false_negative_mask_count: int


@dataclass(frozen=True)
class PrototypeLossOutput:
    loss: torch.Tensor
    query_loss: torch.Tensor
    sample_loss: torch.Tensor
    valid_words_per_sample: torch.Tensor
    valid_word_count: int
    valid_sample_count: int
    ignored_non_master_count: int
    ignored_unavailable_target_count: int
    available_prototype_count: int


@dataclass(frozen=True)
class ContextOTLossOutput:
    total: torch.Tensor
    ot_context: torch.Tensor
    context_token: torch.Tensor
    prototype: torch.Tensor
    weighted_ot: torch.Tensor
    weighted_token: torch.Tensor
    weighted_prototype: torch.Tensor
    ot_output: OTContextLossOutput
    token_output: ContextTokenLossOutput
    prototype_output: PrototypeLossOutput
    diagnostics: dict[str, Any]
