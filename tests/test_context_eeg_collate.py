from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from eeg_keyword_decoding.data import (
    ContextEEGSample,
    collate_context_eeg_samples,
)


def _sample(
    *,
    item: int,
    eeg_length: int,
    word_length: int,
    channels: int = 2,
    context_tail: tuple[int, ...] | None = (3,),
) -> ContextEEGSample:
    context = (
        torch.full(
            (word_length, *context_tail),
            float(item + 1),
            dtype=torch.float32,
        )
        if context_tail is not None
        else None
    )
    return ContextEEGSample(
        eeg_view_id=f"view-{item}",
        subject=f"sub-{item + 1:02d}",
        subject_index=item,
        session="ses",
        task="lis",
        run=item + 1,
        local_row_idx=item,
        global_row_idx=item,
        text_embedding_idx=100 + item,
        sentence_group_id=f"group-{item}",
        outer_fold=0,
        role="train" if context is not None else "validation",
        sfreq=250.0,
        start_sample=10,
        stop_sample=10 + eeg_length,
        eeg=torch.full(
            (channels, eeg_length),
            float(item + 1),
            dtype=torch.float32,
        ),
        eeg_length=eeg_length,
        word_occurrence_ids=tuple(
            f"word-{item}-{position}" for position in range(word_length)
        ),
        word_positions=torch.arange(word_length, dtype=torch.int64),
        word_surface_forms=tuple(
            f"词{position}" for position in range(word_length)
        ),
        word_char_spans=torch.tensor(
            [[position, position + 1] for position in range(word_length)],
            dtype=torch.int64,
        ),
        word_keyword_ids=tuple(
            "kw" if position == 0 else "" for position in range(word_length)
        ),
        word_keyword_indices=torch.tensor(
            [0] + [-1] * (word_length - 1),
            dtype=torch.int64,
        ),
        present_keyword_indices=torch.tensor([0], dtype=torch.int64),
        context_words=context,
        context_backend="backend",
        context_cache_metadata_sha256=(
            "a" * 64 if context is not None else None
        ),
        context_vectors_sha256=(
            "b" * 64 if context is not None else None
        ),
    )


def test_collate_dynamically_pads_both_sequences_and_preserves_metadata():
    first = _sample(item=0, eeg_length=3, word_length=2)
    second = _sample(item=1, eeg_length=5, word_length=4)
    batch = collate_context_eeg_samples([first, second])

    assert batch.eeg.shape == (2, 2, 5)
    assert batch.context_words is not None
    assert batch.context_words.shape == (2, 4, 3)
    assert batch.eeg_mask.dtype == torch.bool
    assert batch.word_mask.dtype == torch.bool
    assert batch.eeg_lengths.tolist() == [3, 5]
    assert batch.word_lengths.tolist() == [2, 4]
    assert batch.eeg_mask.sum(dim=1).tolist() == [3, 5]
    assert batch.word_mask.sum(dim=1).tolist() == [2, 4]
    assert torch.count_nonzero(batch.eeg[0, :, 3:]) == 0
    assert torch.count_nonzero(batch.context_words[0, 2:]) == 0
    assert batch.word_keyword_indices[0, 2:].tolist() == [-1, -1]
    assert batch.word_positions[0, 2:].tolist() == [-1, -1]
    assert batch.word_char_spans[0, 2:].tolist() == [[-1, -1], [-1, -1]]
    assert batch.eeg_view_ids == ("view-0", "view-1")
    assert batch.sentence_group_ids == ("group-0", "group-1")
    assert batch.text_embedding_indices.tolist() == [100, 101]


def test_collate_validation_batch_uses_none_not_fake_context_vectors():
    samples = [
        _sample(
            item=0,
            eeg_length=3,
            word_length=2,
            context_tail=None,
        ),
        _sample(
            item=1,
            eeg_length=4,
            word_length=1,
            context_tail=None,
        ),
    ]
    batch = collate_context_eeg_samples(samples)
    assert batch.context_words is None
    assert batch.context_cache_metadata_sha256 is None
    assert batch.context_vectors_sha256 is None
    assert batch.word_mask.sum(dim=1).tolist() == [2, 1]


def test_batch_to_moves_tensors_but_preserves_string_metadata():
    batch = collate_context_eeg_samples(
        [_sample(item=0, eeg_length=3, word_length=2)]
    )
    moved = batch.to("cpu")
    assert moved.eeg.device.type == "cpu"
    assert moved.eeg_view_ids == batch.eeg_view_ids
    assert moved.word_surface_forms == batch.word_surface_forms


def test_collate_rejects_different_channel_counts():
    with pytest.raises(ValueError, match="channel counts"):
        collate_context_eeg_samples(
            [
                _sample(item=0, eeg_length=3, word_length=2, channels=2),
                _sample(item=1, eeg_length=3, word_length=2, channels=3),
            ]
        )


def test_collate_rejects_different_context_feature_tail_shapes():
    with pytest.raises(ValueError, match="tail shapes"):
        collate_context_eeg_samples(
            [
                _sample(
                    item=0,
                    eeg_length=3,
                    word_length=2,
                    context_tail=(4, 3),
                ),
                _sample(
                    item=1,
                    eeg_length=3,
                    word_length=2,
                    context_tail=(5,),
                ),
            ]
        )


def test_collate_rejects_empty_eeg_and_word_sequences():
    sample = _sample(item=0, eeg_length=3, word_length=2)
    with pytest.raises(ValueError, match="Empty EEG"):
        collate_context_eeg_samples(
            [replace(sample, eeg=torch.empty((2, 0)), eeg_length=0)]
        )
    with pytest.raises(ValueError, match="Empty word"):
        collate_context_eeg_samples(
            [
                replace(
                    sample,
                    word_occurrence_ids=(),
                    word_positions=torch.empty((0,), dtype=torch.int64),
                    word_surface_forms=(),
                    word_char_spans=torch.empty((0, 2), dtype=torch.int64),
                    word_keyword_ids=(),
                    word_keyword_indices=torch.empty(
                        (0,), dtype=torch.int64
                    ),
                    context_words=torch.empty((0, 3)),
                )
            ]
        )


def test_collate_rejects_mixed_context_presence():
    with pytest.raises(ValueError, match="with and without"):
        collate_context_eeg_samples(
            [
                _sample(item=0, eeg_length=3, word_length=2),
                _sample(
                    item=1,
                    eeg_length=3,
                    word_length=2,
                    context_tail=None,
                ),
            ]
        )
