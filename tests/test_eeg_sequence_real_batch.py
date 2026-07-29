from __future__ import annotations

from pathlib import Path

import pytest
import torch

from eeg_keyword_decoding.data import ContextEEGDataset, ContextEEGCollator
from eeg_keyword_decoding.models import build_eeg_sequence_encoder

from conftest import PROJECT_ROOT, PROTOCOL_ROOT


def test_real_inner_train_batch_runs_through_sequence_encoder(
    split_index,
    subject_index,
    keyword_index,
):
    first_record = split_index.records_for(0, "train")[0]
    if not first_record.eeg_vhdr_path.is_file():
        pytest.skip("Local ChineseEEG-2 BrainVision files are unavailable")
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
        include_context_targets=False,
        expected_eeg_channels=128,
    )
    batch = ContextEEGCollator()([dataset[0], dataset[1]])
    model = build_eeg_sequence_encoder(
        PROJECT_ROOT
        / "configs"
        / "models"
        / "eeg_sequence_conv_v1.yaml",
        actual_input_channels=batch.eeg.shape[1],
        actual_num_subjects=len(subject_index.subjects),
    ).eval()
    output = model(
        eeg=batch.eeg,
        eeg_mask=batch.eeg_mask,
        subject_indices=batch.subject_indices,
    )
    assert batch.eeg.dtype == torch.float32
    assert output.sequence.shape == (
        2,
        (batch.eeg.shape[-1] + 3) // 4,
        256,
    )
    assert output.lengths.tolist() == [
        (length + 3) // 4 for length in batch.eeg_lengths.tolist()
    ]
    assert torch.count_nonzero(output.sequence[~output.mask]) == 0
