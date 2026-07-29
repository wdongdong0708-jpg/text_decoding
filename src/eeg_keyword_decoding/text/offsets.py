from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .schema import ContextWordOccurrence


AGGREGATION_RULE = "character_overlap_length_weighted_mean"


class OffsetAlignmentError(ValueError):
    """Raised when a frozen word span cannot be covered by valid subtokens."""


@dataclass(frozen=True)
class TokenAlignment:
    word_occurrence_id: str
    token_indices: tuple[int, ...]
    overlap_lengths: tuple[int, ...]

    @property
    def total_overlap(self) -> int:
        return sum(self.overlap_lengths)


def align_word_to_tokens(
    occurrence: ContextWordOccurrence,
    offset_mapping: Sequence[Sequence[int]],
    special_tokens_mask: Sequence[int],
    attention_mask: Sequence[int] | None = None,
) -> TokenAlignment:
    if len(offset_mapping) != len(special_tokens_mask):
        raise ValueError("offset_mapping and special_tokens_mask length mismatch")
    if attention_mask is not None and len(attention_mask) != len(offset_mapping):
        raise ValueError("attention_mask and offset_mapping length mismatch")

    word_start = occurrence.char_start
    word_end = occurrence.char_end
    token_indices: list[int] = []
    overlap_lengths: list[int] = []
    covered_characters: set[int] = set()

    for token_index, raw_span in enumerate(offset_mapping):
        token_start, token_end = (int(raw_span[0]), int(raw_span[1]))
        attended = attention_mask is None or int(attention_mask[token_index]) == 1
        if not attended:
            continue
        overlap = max(
            0,
            min(word_end, token_end) - max(word_start, token_start),
        )
        if overlap == 0:
            continue
        if int(special_tokens_mask[token_index]) == 1:
            raise OffsetAlignmentError(
                f"Special token overlaps {occurrence.word_occurrence_id}"
            )
        if token_end <= token_start:
            raise OffsetAlignmentError(
                f"Invalid token offset overlaps {occurrence.word_occurrence_id}"
            )
        token_indices.append(token_index)
        overlap_lengths.append(overlap)
        covered_characters.update(
            range(max(word_start, token_start), min(word_end, token_end))
        )

    expected_characters = set(range(word_start, word_end))
    if not token_indices:
        raise OffsetAlignmentError(
            f"No valid subtoken for {occurrence.word_occurrence_id}"
        )
    if covered_characters != expected_characters:
        missing = sorted(expected_characters - covered_characters)
        raise OffsetAlignmentError(
            f"Incomplete subtoken coverage for {occurrence.word_occurrence_id}; "
            f"missing character positions {missing}"
        )
    return TokenAlignment(
        word_occurrence_id=occurrence.word_occurrence_id,
        token_indices=tuple(token_indices),
        overlap_lengths=tuple(overlap_lengths),
    )


def align_sentence_to_tokens(
    occurrences: Sequence[ContextWordOccurrence],
    offset_mapping: Sequence[Sequence[int]],
    special_tokens_mask: Sequence[int],
    attention_mask: Sequence[int] | None = None,
) -> tuple[TokenAlignment, ...]:
    return tuple(
        align_word_to_tokens(
            occurrence,
            offset_mapping,
            special_tokens_mask,
            attention_mask,
        )
        for occurrence in occurrences
    )


def aggregate_token_features(
    token_features: np.ndarray,
    alignments: Sequence[TokenAlignment],
) -> np.ndarray:
    """Aggregate ``[tokens, ..., hidden]`` features into frozen word rows."""

    features = np.asarray(token_features)
    if features.ndim < 2:
        raise ValueError("token_features must have at least two dimensions")
    outputs: list[np.ndarray] = []
    for alignment in alignments:
        indices = np.asarray(alignment.token_indices, dtype=np.int64)
        weights = np.asarray(alignment.overlap_lengths, dtype=np.float64)
        if indices.size == 0 or weights.sum() <= 0:
            raise ValueError(
                f"Empty alignment for {alignment.word_occurrence_id}"
            )
        if indices.min() < 0 or indices.max() >= features.shape[0]:
            raise IndexError(
                f"Token index outside feature matrix for "
                f"{alignment.word_occurrence_id}"
            )
        normalized_weights = weights / weights.sum()
        aggregated = np.tensordot(
            normalized_weights,
            features[indices].astype(np.float64, copy=False),
            axes=(0, 0),
        )
        outputs.append(aggregated.astype(features.dtype, copy=False))
    if not outputs:
        return np.empty((0, *features.shape[1:]), dtype=features.dtype)
    return np.stack(outputs, axis=0)
