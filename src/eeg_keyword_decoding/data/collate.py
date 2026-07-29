from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .sample import ContextEEGBatch, ContextEEGSample


def _single_value(values: Sequence[object], name: str) -> object:
    unique = set(values)
    if len(unique) != 1:
        raise ValueError(f"Batch mixes incompatible {name}: {unique}")
    return values[0]


def collate_context_eeg_samples(
    samples: Sequence[ContextEEGSample],
    *,
    pin_memory: bool = False,
) -> ContextEEGBatch:
    if not samples:
        raise ValueError("Cannot collate an empty sample list")

    batch_size = len(samples)
    channel_counts: list[int] = []
    eeg_lengths: list[int] = []
    word_lengths: list[int] = []
    context_presence: list[bool] = []
    context_tail_shapes: list[tuple[int, ...]] = []
    for sample in samples:
        if sample.eeg.ndim != 2:
            raise ValueError(
                f"EEG must have shape [channel, time], got {sample.eeg.shape}"
            )
        if sample.eeg.dtype != torch.float32:
            raise ValueError(
                f"EEG must be float32, got {sample.eeg.dtype}"
            )
        channels, eeg_length = sample.eeg.shape
        if channels <= 0 or eeg_length <= 0:
            raise ValueError("Empty EEG sequences cannot be collated")
        if eeg_length != sample.eeg_length:
            raise ValueError(
                f"eeg_length disagrees with tensor shape for "
                f"{sample.eeg_view_id}"
            )
        word_length = len(sample.word_occurrence_ids)
        if word_length <= 0:
            raise ValueError("Empty word sequences cannot be collated")
        if sample.word_positions.shape != (word_length,):
            raise ValueError("word_positions shape mismatch")
        if sample.word_keyword_indices.shape != (word_length,):
            raise ValueError("word_keyword_indices shape mismatch")
        if sample.word_char_spans.shape != (word_length, 2):
            raise ValueError("word_char_spans shape mismatch")
        if len(sample.word_surface_forms) != word_length:
            raise ValueError("word_surface_forms length mismatch")
        if len(sample.word_keyword_ids) != word_length:
            raise ValueError("word_keyword_ids length mismatch")
        if sample.word_positions.dtype != torch.int64:
            raise ValueError("word_positions must be int64")
        if sample.word_keyword_indices.dtype != torch.int64:
            raise ValueError("word_keyword_indices must be int64")
        if sample.word_char_spans.dtype != torch.int64:
            raise ValueError("word_char_spans must be int64")
        if sample.present_keyword_indices.dtype != torch.int64:
            raise ValueError("present_keyword_indices must be int64")

        has_context = sample.context_words is not None
        if has_context:
            assert sample.context_words is not None
            if sample.context_words.ndim < 2:
                raise ValueError(
                    "context_words must have [word, ...feature] axes"
                )
            if sample.context_words.shape[0] != word_length:
                raise ValueError("context_words word count mismatch")
            if sample.context_words.dtype != torch.float32:
                raise ValueError(
                    "Context targets must be restored as float32"
                )
            context_tail_shapes.append(
                tuple(sample.context_words.shape[1:])
            )
        channel_counts.append(channels)
        eeg_lengths.append(eeg_length)
        word_lengths.append(word_length)
        context_presence.append(has_context)

    if len(set(channel_counts)) != 1:
        raise ValueError(
            f"Batch contains different EEG channel counts: {channel_counts}"
        )
    if len(set(context_presence)) != 1:
        raise ValueError(
            "Batch cannot mix samples with and without context targets"
        )
    if context_tail_shapes and len(set(context_tail_shapes)) != 1:
        raise ValueError(
            "Batch contains different context feature tail shapes: "
            f"{context_tail_shapes}"
        )

    context_backend = _single_value(
        [sample.context_backend for sample in samples],
        "context backends",
    )
    context_metadata_hash = _single_value(
        [
            sample.context_cache_metadata_sha256
            for sample in samples
        ],
        "context cache metadata hashes",
    )
    context_vectors_hash = _single_value(
        [sample.context_vectors_sha256 for sample in samples],
        "context vector hashes",
    )
    _single_value([sample.role for sample in samples], "split roles")
    _single_value(
        [sample.outer_fold for sample in samples],
        "outer folds",
    )

    channels = channel_counts[0]
    maximum_eeg_length = max(eeg_lengths)
    maximum_word_length = max(word_lengths)
    eeg = torch.zeros(
        (batch_size, channels, maximum_eeg_length),
        dtype=torch.float32,
    )
    eeg_mask = torch.zeros(
        (batch_size, maximum_eeg_length),
        dtype=torch.bool,
    )
    word_mask = torch.zeros(
        (batch_size, maximum_word_length),
        dtype=torch.bool,
    )
    word_keyword_indices = torch.full(
        (batch_size, maximum_word_length),
        fill_value=-1,
        dtype=torch.int64,
    )
    word_positions = torch.full(
        (batch_size, maximum_word_length),
        fill_value=-1,
        dtype=torch.int64,
    )
    word_char_spans = torch.full(
        (batch_size, maximum_word_length, 2),
        fill_value=-1,
        dtype=torch.int64,
    )
    context_words: torch.Tensor | None = None
    if context_tail_shapes:
        context_words = torch.zeros(
            (
                batch_size,
                maximum_word_length,
                *context_tail_shapes[0],
            ),
            dtype=torch.float32,
        )

    for batch_index, sample in enumerate(samples):
        eeg_length = eeg_lengths[batch_index]
        word_length = word_lengths[batch_index]
        eeg[batch_index, :, :eeg_length] = sample.eeg
        eeg_mask[batch_index, :eeg_length] = True
        word_mask[batch_index, :word_length] = True
        word_keyword_indices[
            batch_index, :word_length
        ] = sample.word_keyword_indices
        word_positions[batch_index, :word_length] = sample.word_positions
        word_char_spans[
            batch_index, :word_length
        ] = sample.word_char_spans
        if context_words is not None:
            assert sample.context_words is not None
            context_words[
                batch_index, :word_length
            ] = sample.context_words

    eeg_lengths_tensor = torch.tensor(eeg_lengths, dtype=torch.int64)
    word_lengths_tensor = torch.tensor(word_lengths, dtype=torch.int64)
    if not torch.equal(eeg_mask.sum(dim=1), eeg_lengths_tensor):
        raise RuntimeError("Internal eeg_mask/eeg_lengths mismatch")
    if not torch.equal(word_mask.sum(dim=1), word_lengths_tensor):
        raise RuntimeError("Internal word_mask/word_lengths mismatch")

    batch = ContextEEGBatch(
        eeg=eeg,
        eeg_mask=eeg_mask,
        eeg_lengths=eeg_lengths_tensor,
        context_words=context_words,
        word_mask=word_mask,
        word_lengths=word_lengths_tensor,
        word_keyword_indices=word_keyword_indices,
        word_positions=word_positions,
        word_char_spans=word_char_spans,
        subject_indices=torch.tensor(
            [sample.subject_index for sample in samples],
            dtype=torch.int64,
        ),
        text_embedding_indices=torch.tensor(
            [sample.text_embedding_idx for sample in samples],
            dtype=torch.int64,
        ),
        outer_folds=torch.tensor(
            [sample.outer_fold for sample in samples],
            dtype=torch.int64,
        ),
        runs=torch.tensor(
            [sample.run for sample in samples],
            dtype=torch.int64,
        ),
        sfreqs=torch.tensor(
            [sample.sfreq for sample in samples],
            dtype=torch.float32,
        ),
        present_keyword_indices=tuple(
            sample.present_keyword_indices for sample in samples
        ),
        eeg_view_ids=tuple(sample.eeg_view_id for sample in samples),
        subjects=tuple(sample.subject for sample in samples),
        sessions=tuple(sample.session for sample in samples),
        tasks=tuple(sample.task for sample in samples),
        roles=tuple(sample.role for sample in samples),
        sentence_group_ids=tuple(
            sample.sentence_group_id for sample in samples
        ),
        word_occurrence_ids=tuple(
            sample.word_occurrence_ids for sample in samples
        ),
        word_surface_forms=tuple(
            sample.word_surface_forms for sample in samples
        ),
        word_keyword_ids=tuple(
            sample.word_keyword_ids for sample in samples
        ),
        context_backend=(
            str(context_backend)
            if context_backend is not None
            else None
        ),
        context_cache_metadata_sha256=(
            str(context_metadata_hash)
            if context_metadata_hash is not None
            else None
        ),
        context_vectors_sha256=(
            str(context_vectors_hash)
            if context_vectors_hash is not None
            else None
        ),
    )
    return batch.pin_memory() if pin_memory else batch


@dataclass(frozen=True)
class ContextEEGCollator:
    pin_memory: bool = False

    def __call__(
        self,
        samples: Sequence[ContextEEGSample],
    ) -> ContextEEGBatch:
        return collate_context_eeg_samples(
            samples,
            pin_memory=self.pin_memory,
        )
