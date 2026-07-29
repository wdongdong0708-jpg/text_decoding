from __future__ import annotations

import torch
from torch.nn import functional as F

from eeg_keyword_decoding.prototypes import PrototypeBank

from .outputs import PrototypeLossOutput


def prototype_classification_loss(
    *,
    word_conditioned_eeg: torch.Tensor,
    word_keyword_indices: torch.Tensor,
    word_mask: torch.Tensor,
    prototype_bank: PrototypeBank,
    temperature: float = 0.07,
) -> PrototypeLossOutput:
    if temperature <= 0:
        raise ValueError("Prototype temperature must be positive")
    prototype_bank.validate()
    if word_conditioned_eeg.ndim != 3:
        raise ValueError(
            "word_conditioned_eeg must have shape [batch,word,feature]"
        )
    batch_size, word_count, feature_dim = word_conditioned_eeg.shape
    if feature_dim != prototype_bank.vectors.shape[1]:
        raise ValueError("EEG/prototype dimensions differ")
    if word_keyword_indices.shape != (batch_size, word_count):
        raise ValueError("word_keyword_indices shape mismatch")
    if word_mask.shape != (batch_size, word_count):
        raise ValueError("word_mask shape mismatch")
    if word_mask.dtype != torch.bool:
        raise ValueError("word_mask must be torch.bool")
    if word_keyword_indices.dtype != torch.int64:
        raise ValueError("word_keyword_indices must be int64")
    devices = {
        word_conditioned_eeg.device,
        word_keyword_indices.device,
        word_mask.device,
        prototype_bank.vectors.device,
        prototype_bank.available_mask.device,
    }
    if len(devices) != 1:
        raise ValueError("Prototype loss inputs must share a device")
    if not bool(torch.isfinite(word_conditioned_eeg).all()):
        raise ValueError("EEG word representations contain NaN or Inf")

    master_mask = word_mask & (word_keyword_indices >= 0)
    if bool((word_keyword_indices[master_mask] >= 247).any()):
        raise ValueError("Master keyword index is out of range")
    target_available = torch.zeros_like(word_mask)
    if bool(master_mask.any()):
        target_available[master_mask] = prototype_bank.available_mask[
            word_keyword_indices[master_mask]
        ]
    valid = master_mask & target_available
    unavailable_target_count = int(
        (master_mask & ~target_available).sum().item()
    )
    non_master_count = int((word_mask & ~master_mask).sum().item())
    query_loss = torch.zeros(
        (batch_size, word_count),
        dtype=torch.float32,
        device=word_conditioned_eeg.device,
    )
    sample_loss = torch.zeros(
        batch_size,
        dtype=torch.float32,
        device=word_conditioned_eeg.device,
    )
    valid_counts = valid.sum(dim=1, dtype=torch.int64)

    if bool(valid.any()):
        eeg = F.normalize(
            word_conditioned_eeg[valid].float(),
            p=2,
            dim=-1,
            eps=1e-8,
        )
        # The bank is a detached snapshot. Gradients still reach the EEG word
        # path, including the current OT plan and current text projector.
        prototypes = F.normalize(
            prototype_bank.vectors.detach().float(),
            p=2,
            dim=-1,
            eps=1e-8,
        )
        logits = eeg @ prototypes.transpose(0, 1) / float(temperature)
        logits = logits.masked_fill(
            ~prototype_bank.available_mask.unsqueeze(0),
            -torch.inf,
        )
        targets = word_keyword_indices[valid]
        flat_loss = F.cross_entropy(logits, targets, reduction="none")
        query_loss = query_loss.masked_scatter(valid, flat_loss)
        sums = query_loss.sum(dim=1)
        sample_loss = torch.where(
            valid_counts > 0,
            sums / valid_counts.clamp_min(1).to(sums.dtype),
            torch.zeros_like(sums),
        )
        contributing_samples = valid_counts > 0
        loss = sample_loss[contributing_samples].mean()
    else:
        loss = word_conditioned_eeg.sum() * 0.0

    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Prototype loss is NaN or Inf")
    return PrototypeLossOutput(
        loss=loss,
        query_loss=query_loss,
        sample_loss=sample_loss,
        valid_words_per_sample=valid_counts,
        valid_word_count=int(valid.sum().item()),
        valid_sample_count=int((valid_counts > 0).sum().item()),
        ignored_non_master_count=non_master_count,
        ignored_unavailable_target_count=unavailable_target_count,
        available_prototype_count=prototype_bank.available_count,
    )
