from __future__ import annotations

import torch
from torch import nn

from eeg_keyword_decoding.models import ContextTextProjector

from .costs import masked_cosine_cost
from .outputs import ContextOTOutput
from .sinkhorn import MaskedBalancedSinkhorn
from .transport import transport_pool_eeg_to_words


class ContextOTAligner(nn.Module):
    """Text projection, cosine cost, balanced OT, and word pooling."""

    def __init__(
        self,
        *,
        text_projector: ContextTextProjector,
        sinkhorn: MaskedBalancedSinkhorn,
        cosine_eps: float = 1e-8,
        transport_eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if cosine_eps <= 0 or transport_eps <= 0:
            raise ValueError("Alignment eps values must be positive")
        self.text_projector = text_projector
        self.sinkhorn = sinkhorn
        self.cosine_eps = float(cosine_eps)
        self.transport_eps = float(transport_eps)

    def forward(
        self,
        *,
        eeg_sequence: torch.Tensor,
        eeg_mask: torch.Tensor,
        context_words: torch.Tensor,
        word_mask: torch.Tensor,
        context_backend: str,
    ) -> ContextOTOutput:
        text = self.text_projector(
            context_words=context_words,
            word_mask=word_mask,
            backend=context_backend,
        )
        cost = masked_cosine_cost(
            eeg_sequence=eeg_sequence,
            text_sequence=text.sequence,
            eeg_mask=eeg_mask,
            word_mask=text.mask,
            eps=self.cosine_eps,
        )
        sinkhorn = self.sinkhorn(
            cost=cost.cost,
            eeg_mask=eeg_mask,
            word_mask=text.mask,
            valid_pair_mask=cost.valid_pair_mask,
        )
        pooled = transport_pool_eeg_to_words(
            eeg_sequence=eeg_sequence,
            plan=sinkhorn.plan,
            word_mask=text.mask,
            eps=self.transport_eps,
        )
        return ContextOTOutput(
            text=text,
            cost=cost,
            sinkhorn=sinkhorn,
            word_conditioned_eeg=pooled,
        )
