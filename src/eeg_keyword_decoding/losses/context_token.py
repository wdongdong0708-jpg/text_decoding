from __future__ import annotations

import torch
from torch.nn import functional as F

from .outputs import ContextTokenLossOutput


def _validate_inputs(
    *,
    eeg_words: torch.Tensor,
    text_words: torch.Tensor,
    word_mask: torch.Tensor,
    context_token_group_indices: torch.Tensor,
    surface_type_indices: torch.Tensor,
    sentence_group_indices: torch.Tensor,
) -> None:
    if eeg_words.ndim != 3 or text_words.ndim != 3:
        raise ValueError("Token sequences must have shape [batch,word,feature]")
    if eeg_words.shape != text_words.shape:
        raise ValueError("EEG and text token sequence shapes differ")
    batch_size, word_count, _ = eeg_words.shape
    expected = (batch_size, word_count)
    for name, value in (
        ("word_mask", word_mask),
        ("context_token_group_indices", context_token_group_indices),
        ("surface_type_indices", surface_type_indices),
    ):
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
    if word_mask.dtype != torch.bool:
        raise ValueError("word_mask must be torch.bool")
    if (
        context_token_group_indices.dtype != torch.int64
        or surface_type_indices.dtype != torch.int64
        or sentence_group_indices.dtype != torch.int64
    ):
        raise ValueError("Identity tensors must be int64")
    if sentence_group_indices.shape != (batch_size,):
        raise ValueError("sentence_group_indices must have shape [batch]")
    devices = {
        eeg_words.device,
        text_words.device,
        word_mask.device,
        context_token_group_indices.device,
        surface_type_indices.device,
        sentence_group_indices.device,
    }
    if len(devices) != 1:
        raise ValueError("Context-token inputs must share a device")
    if not bool(torch.isfinite(eeg_words).all()):
        raise ValueError("EEG word representations contain NaN or Inf")
    if not bool(torch.isfinite(text_words).all()):
        raise ValueError("Text word representations contain NaN or Inf")
    if bool((word_mask.sum(dim=1) == 0).any()):
        raise ValueError("Every sample must contain at least one valid word")
    if bool((context_token_group_indices[word_mask] < 0).any()):
        raise ValueError("Valid words require context-token group indices")
    if bool((surface_type_indices[word_mask] < 0).any()):
        raise ValueError("Valid words require lexical surface indices")


def _sample_balanced(
    query_loss: torch.Tensor,
    sample_indices: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sums = torch.zeros(
        batch_size,
        dtype=query_loss.dtype,
        device=query_loss.device,
    ).index_add(0, sample_indices, query_loss)
    counts = torch.bincount(
        sample_indices,
        minlength=batch_size,
    ).to(query_loss.dtype)
    valid = counts > 0
    sample_loss = torch.where(
        valid,
        sums / counts.clamp_min(1.0),
        torch.zeros_like(sums),
    )
    if not bool(valid.any()):
        return query_loss.sum() * 0.0, sample_loss, counts.to(torch.int64)
    return sample_loss[valid].mean(), sample_loss, counts.to(torch.int64)


def _group_balanced_sample_mean(
    sample_loss: torch.Tensor,
    sentence_group_indices: torch.Tensor,
    valid_sample_mask: torch.Tensor,
) -> torch.Tensor:
    if not bool(valid_sample_mask.any()):
        return sample_loss.sum() * 0.0
    sample_loss = sample_loss[valid_sample_mask]
    sentence_group_indices = sentence_group_indices[valid_sample_mask]
    unique, inverse = torch.unique(
        sentence_group_indices,
        sorted=True,
        return_inverse=True,
    )
    sums = torch.zeros(
        unique.numel(),
        dtype=sample_loss.dtype,
        device=sample_loss.device,
    ).index_add(0, inverse, sample_loss)
    counts = torch.bincount(
        inverse,
        minlength=unique.numel(),
    ).to(sample_loss.dtype)
    return (sums / counts.clamp_min(1.0)).mean()


def _direction_loss(
    logits: torch.Tensor,
    *,
    positive_mask: torch.Tensor,
    denominator_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    positive_counts = positive_mask.sum(dim=1, dtype=torch.int64)
    if bool((positive_counts == 0).any()):
        raise ValueError("Every context-token query requires a positive")
    negative_mask = denominator_mask & ~positive_mask
    negative_counts = negative_mask.sum(dim=1, dtype=torch.int64)
    positive_log_mass = torch.logsumexp(
        logits.masked_fill(~positive_mask, -torch.inf),
        dim=1,
    )
    denominator_log_mass = torch.logsumexp(
        logits.masked_fill(~denominator_mask, -torch.inf),
        dim=1,
    )
    has_negative = negative_counts > 0
    query_loss = torch.where(
        has_negative,
        denominator_log_mass - positive_log_mass,
        torch.zeros_like(denominator_log_mass),
    )
    if not bool(torch.isfinite(query_loss).all()):
        raise FloatingPointError("Context-token InfoNCE produced NaN or Inf")
    return query_loss, positive_counts, negative_counts, has_negative


def context_token_info_nce(
    *,
    word_conditioned_eeg: torch.Tensor,
    projected_context_words: torch.Tensor,
    word_mask: torch.Tensor,
    context_token_group_indices: torch.Tensor,
    surface_type_indices: torch.Tensor,
    sentence_group_indices: torch.Tensor,
    temperature: float = 0.07,
    symmetric: bool = True,
) -> ContextTokenLossOutput:
    """Symmetric multi-positive InfoNCE with lexical false-negative masking."""

    if temperature <= 0:
        raise ValueError("Context-token temperature must be positive")
    _validate_inputs(
        eeg_words=word_conditioned_eeg,
        text_words=projected_context_words,
        word_mask=word_mask,
        context_token_group_indices=context_token_group_indices,
        surface_type_indices=surface_type_indices,
        sentence_group_indices=sentence_group_indices,
    )
    batch_size, word_count, _ = word_conditioned_eeg.shape
    sample_grid = (
        torch.arange(batch_size, device=word_mask.device)
        .unsqueeze(1)
        .expand(batch_size, word_count)
    )
    sample_indices = sample_grid[word_mask]
    context_groups = context_token_group_indices[word_mask]
    surface_types = surface_type_indices[word_mask]
    eeg = F.normalize(
        word_conditioned_eeg[word_mask].float(),
        p=2,
        dim=-1,
        eps=1e-8,
    )
    text = F.normalize(
        projected_context_words[word_mask].float(),
        p=2,
        dim=-1,
        eps=1e-8,
    )
    logits = eeg @ text.transpose(0, 1) / float(temperature)

    same_context = context_groups[:, None].eq(context_groups[None, :])
    same_surface = surface_types[:, None].eq(surface_types[None, :])
    if bool((same_context & ~same_surface).any()):
        raise ValueError(
            "A context-token group contains different lexical surfaces"
        )
    false_negative_mask = same_surface & ~same_context
    denominator_mask = ~false_negative_mask

    e2t_query, positive_counts, negative_counts, e2t_contributing = (
        _direction_loss(
            logits,
            positive_mask=same_context,
            denominator_mask=denominator_mask,
        )
    )
    e2t_loss, e2t_sample, _ = _sample_balanced(
        e2t_query[e2t_contributing],
        sample_indices[e2t_contributing],
        batch_size=batch_size,
    )
    valid_words = torch.bincount(
        sample_indices,
        minlength=batch_size,
    ).to(torch.int64)

    t2e_query, _, _, t2e_contributing = _direction_loss(
        logits.transpose(0, 1),
        positive_mask=same_context,
        denominator_mask=denominator_mask,
    )
    _, t2e_sample, t2e_contributing_words = _sample_balanced(
        t2e_query[t2e_contributing],
        sample_indices[t2e_contributing],
        batch_size=batch_size,
    )
    # Duplicate normalized contexts may appear through multiple EEG views.
    # Collapse their text-query sample losses before the direction mean.
    t2e_loss = _group_balanced_sample_mean(
        t2e_sample,
        sentence_group_indices,
        t2e_contributing_words > 0,
    )
    loss = 0.5 * (e2t_loss + t2e_loss) if symmetric else e2t_loss
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Context-token loss is NaN or Inf")
    return ContextTokenLossOutput(
        loss=loss,
        eeg_to_text_loss=e2t_loss,
        text_to_eeg_loss=t2e_loss,
        eeg_to_text_query_loss=e2t_query,
        text_to_eeg_query_loss=t2e_query,
        eeg_to_text_sample_loss=e2t_sample,
        text_to_eeg_sample_loss=t2e_sample,
        valid_words_per_sample=valid_words,
        positive_counts=positive_counts,
        negative_counts=negative_counts,
        valid_query_count=int(e2t_query.numel()),
        contributing_query_count=int(
            (e2t_contributing | t2e_contributing).sum().item()
        ),
        false_negative_mask_count=int(false_negative_mask.sum().item()),
    )
