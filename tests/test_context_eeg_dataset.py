from __future__ import annotations

import numpy as np
import pytest
import torch

from eeg_keyword_decoding.data import ContextEEGDataset

from conftest import PROTOCOL_ROOT


class SyntheticEEGReader:
    n_channels = 128
    sfreq = 250.0

    def __init__(self, _path):
        pass

    def read_window(self, start_sample: int, stop_sample: int) -> np.ndarray:
        length = stop_sample - start_sample
        return np.broadcast_to(
            np.arange(length, dtype=np.float32),
            (self.n_channels, length),
        )


class WrongChannelEEGReader(SyntheticEEGReader):
    n_channels = 127


def _dataset(
    *,
    split_index,
    subject_index,
    keyword_index,
    role="train",
    backend="bge_m3",
    context_store=None,
    include_context_targets=None,
    reader_factory=SyntheticEEGReader,
    expected_cache_metadata_sha256=None,
):
    return ContextEEGDataset(
        split_index=split_index,
        outer_fold=0,
        role=role,
        subject_index=subject_index,
        keyword_index=keyword_index,
        sentence_labels_path=(
            PROTOCOL_ROOT / "littleprince_sentence_keyword_labels_v1.csv"
        ),
        word_occurrences_path=(
            PROTOCOL_ROOT / "littleprince_word_occurrences_v1.csv"
        ),
        text_backend=backend,
        context_store_root=context_store,
        include_context_targets=include_context_targets,
        reader_factory=reader_factory,
        expected_eeg_channels=128,
        expected_cache_metadata_sha256=(
            expected_cache_metadata_sha256
        ),
    )


@pytest.mark.parametrize(
    ("backend", "expected_tail"),
    [("macbert", (4, 3)), ("bge_m3", (5,))],
)
def test_train_dataset_reads_both_context_store_backends(
    split_index,
    subject_index,
    keyword_index,
    synthetic_context_stores,
    backend,
    expected_tail,
):
    dataset = _dataset(
        split_index=split_index,
        subject_index=subject_index,
        keyword_index=keyword_index,
        backend=backend,
        context_store=synthetic_context_stores[backend],
    )
    sample = dataset[0]
    record = dataset.records[0]

    assert len(dataset) == len(split_index.records_for(0, "train"))
    assert sample.eeg.shape == (128, record.n_samples)
    assert sample.eeg.dtype == torch.float32
    assert sample.eeg_length == record.n_samples
    assert sample.context_words is not None
    assert sample.context_words.shape == (
        len(sample.word_occurrence_ids),
        *expected_tail,
    )
    assert sample.context_words.dtype == torch.float32
    assert sample.word_positions.tolist() == list(
        range(len(sample.word_occurrence_ids))
    )
    assert len(sample.word_surface_forms) == len(
        sample.word_occurrence_ids
    )
    assert sample.context_backend == backend
    assert len(sample.context_cache_metadata_sha256 or "") == 64
    assert len(sample.context_vectors_sha256 or "") == 64


def test_dataset_word_count_matches_frozen_fold_token_count(
    split_index,
    subject_index,
    keyword_index,
    synthetic_context_stores,
):
    dataset = _dataset(
        split_index=split_index,
        subject_index=subject_index,
        keyword_index=keyword_index,
        context_store=synthetic_context_stores["bge_m3"],
    )
    sample = dataset[0]
    assignment = split_index.assignment(0, sample.text_embedding_idx)
    assert len(sample.word_occurrence_ids) == assignment.token_count
    assert sample.word_keyword_indices.shape == (assignment.token_count,)
    assert sample.context_token_group_indices.shape == (
        assignment.token_count,
    )
    assert sample.surface_type_indices.shape == (assignment.token_count,)
    assert bool((sample.context_token_group_indices >= 0).all())
    assert bool((sample.surface_type_indices >= 0).all())
    assert all(
        occurrence_id.startswith(f"lp:{sample.text_embedding_idx}:word:")
        for occurrence_id in sample.word_occurrence_ids
    )


def test_dataset_rejects_expected_cache_metadata_hash_mismatch(
    split_index,
    subject_index,
    keyword_index,
    synthetic_context_stores,
):
    with pytest.raises(ValueError, match="metadata SHA256 mismatch"):
        _dataset(
            split_index=split_index,
            subject_index=subject_index,
            keyword_index=keyword_index,
            context_store=synthetic_context_stores["bge_m3"],
            expected_cache_metadata_sha256="0" * 64,
        )


def test_dataset_rejects_channel_count_mismatch(
    split_index,
    subject_index,
    keyword_index,
    synthetic_context_stores,
):
    dataset = _dataset(
        split_index=split_index,
        subject_index=subject_index,
        keyword_index=keyword_index,
        context_store=synthetic_context_stores["bge_m3"],
        reader_factory=WrongChannelEEGReader,
    )
    with pytest.raises(ValueError, match="channel mismatch"):
        _ = dataset[0]


@pytest.mark.parametrize("role", ["validation", "test"])
def test_evaluation_dataset_returns_truth_but_no_context_vectors(
    split_index,
    subject_index,
    keyword_index,
    role,
):
    dataset = _dataset(
        split_index=split_index,
        subject_index=subject_index,
        keyword_index=keyword_index,
        role=role,
        context_store=None,
        include_context_targets=False,
    )
    sample = dataset[0]
    assert sample.context_words is None
    assert sample.context_cache_metadata_sha256 is None
    assert sample.context_vectors_sha256 is None
    assert sample.present_keyword_indices.dtype == torch.int64
    assert len(sample.word_occurrence_ids) > 0
