from __future__ import annotations

from pathlib import Path

import pytest
from torch.utils.data import DataLoader

from eeg_keyword_decoding.data import (
    ContextEEGCollator,
    ContextEEGDataset,
)

from conftest import PROJECT_ROOT, PROTOCOL_ROOT


def test_context_eeg_dataset_works_with_spawned_workers(
    split_index,
    subject_index,
    keyword_index,
):
    first_record = split_index.records_for(0, "train")[0]
    if not first_record.eeg_vhdr_path.is_file():
        pytest.skip("Local ChineseEEG-2 BrainVision files are unavailable")
    cache_root = (
        PROJECT_ROOT
        / "data"
        / "cache"
        / "context_words"
        / "bge_m3_colbert_v1"
    )
    if not cache_root.is_dir():
        pytest.skip("Local BGE-M3 context cache is unavailable")

    dataset = ContextEEGDataset(
        split_index=split_index,
        outer_fold=0,
        role="train",
        subject_index=subject_index,
        keyword_index=keyword_index,
        sentence_labels_path=(
            PROTOCOL_ROOT / "littleprince_sentence_keyword_labels_v1.csv"
        ),
        word_occurrences_path=(
            PROTOCOL_ROOT / "littleprince_word_occurrences_v1.csv"
        ),
        text_backend="bge_m3",
        context_store_root=cache_root,
        expected_eeg_channels=128,
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        multiprocessing_context="spawn",
        collate_fn=ContextEEGCollator(),
    )
    batch = next(iter(loader))
    assert batch.eeg.shape[0] == 2
    assert batch.context_words is not None
    assert batch.context_words.shape[0] == 2
    assert batch.eeg_mask.sum(dim=1).tolist() == batch.eeg_lengths.tolist()
    assert batch.word_mask.sum(dim=1).tolist() == batch.word_lengths.tolist()
