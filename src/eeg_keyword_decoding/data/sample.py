from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch


@dataclass(frozen=True)
class ContextEEGSample:
    eeg_view_id: str
    subject: str
    subject_index: int
    session: str
    task: str
    run: int
    local_row_idx: int
    global_row_idx: int
    text_embedding_idx: int
    sentence_group_id: str
    outer_fold: int
    role: str
    sfreq: float
    start_sample: int
    stop_sample: int
    eeg: torch.Tensor
    eeg_length: int
    word_occurrence_ids: tuple[str, ...]
    word_positions: torch.Tensor
    word_surface_forms: tuple[str, ...]
    word_char_spans: torch.Tensor
    word_keyword_ids: tuple[str, ...]
    word_keyword_indices: torch.Tensor
    present_keyword_indices: torch.Tensor
    context_words: torch.Tensor | None
    context_backend: str | None
    context_cache_metadata_sha256: str | None
    context_vectors_sha256: str | None


@dataclass(frozen=True)
class ContextEEGBatch:
    eeg: torch.Tensor
    eeg_mask: torch.Tensor
    eeg_lengths: torch.Tensor
    context_words: torch.Tensor | None
    word_mask: torch.Tensor
    word_lengths: torch.Tensor
    word_keyword_indices: torch.Tensor
    word_positions: torch.Tensor
    word_char_spans: torch.Tensor
    subject_indices: torch.Tensor
    text_embedding_indices: torch.Tensor
    outer_folds: torch.Tensor
    runs: torch.Tensor
    sfreqs: torch.Tensor
    present_keyword_indices: tuple[torch.Tensor, ...]
    eeg_view_ids: tuple[str, ...]
    subjects: tuple[str, ...]
    sessions: tuple[str, ...]
    tasks: tuple[str, ...]
    roles: tuple[str, ...]
    sentence_group_ids: tuple[str, ...]
    word_occurrence_ids: tuple[tuple[str, ...], ...]
    word_surface_forms: tuple[tuple[str, ...], ...]
    word_keyword_ids: tuple[tuple[str, ...], ...]
    context_backend: str | None
    context_cache_metadata_sha256: str | None
    context_vectors_sha256: str | None

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> ContextEEGBatch:
        target = torch.device(device)

        def move(value: torch.Tensor | None) -> torch.Tensor | None:
            if value is None:
                return None
            return value.to(target, non_blocking=non_blocking)

        return replace(
            self,
            eeg=move(self.eeg),  # type: ignore[arg-type]
            eeg_mask=move(self.eeg_mask),  # type: ignore[arg-type]
            eeg_lengths=move(self.eeg_lengths),  # type: ignore[arg-type]
            context_words=move(self.context_words),
            word_mask=move(self.word_mask),  # type: ignore[arg-type]
            word_lengths=move(self.word_lengths),  # type: ignore[arg-type]
            word_keyword_indices=move(  # type: ignore[arg-type]
                self.word_keyword_indices
            ),
            word_positions=move(self.word_positions),  # type: ignore[arg-type]
            word_char_spans=move(  # type: ignore[arg-type]
                self.word_char_spans
            ),
            subject_indices=move(  # type: ignore[arg-type]
                self.subject_indices
            ),
            text_embedding_indices=move(  # type: ignore[arg-type]
                self.text_embedding_indices
            ),
            outer_folds=move(self.outer_folds),  # type: ignore[arg-type]
            runs=move(self.runs),  # type: ignore[arg-type]
            sfreqs=move(self.sfreqs),  # type: ignore[arg-type]
            present_keyword_indices=tuple(
                value.to(target, non_blocking=non_blocking)
                for value in self.present_keyword_indices
            ),
        )

    def pin_memory(self) -> ContextEEGBatch:
        def pin(value: torch.Tensor | None) -> torch.Tensor | None:
            if value is None:
                return None
            return value.pin_memory()

        return replace(
            self,
            eeg=pin(self.eeg),  # type: ignore[arg-type]
            eeg_mask=pin(self.eeg_mask),  # type: ignore[arg-type]
            eeg_lengths=pin(self.eeg_lengths),  # type: ignore[arg-type]
            context_words=pin(self.context_words),
            word_mask=pin(self.word_mask),  # type: ignore[arg-type]
            word_lengths=pin(self.word_lengths),  # type: ignore[arg-type]
            word_keyword_indices=pin(  # type: ignore[arg-type]
                self.word_keyword_indices
            ),
            word_positions=pin(self.word_positions),  # type: ignore[arg-type]
            word_char_spans=pin(  # type: ignore[arg-type]
                self.word_char_spans
            ),
            subject_indices=pin(  # type: ignore[arg-type]
                self.subject_indices
            ),
            text_embedding_indices=pin(  # type: ignore[arg-type]
                self.text_embedding_indices
            ),
            outer_folds=pin(self.outer_folds),  # type: ignore[arg-type]
            runs=pin(self.runs),  # type: ignore[arg-type]
            sfreqs=pin(self.sfreqs),  # type: ignore[arg-type]
            present_keyword_indices=tuple(
                value.pin_memory() for value in self.present_keyword_indices
            ),
        )

    def tensor_shapes(self) -> dict[str, Any]:
        return {
            "eeg": list(self.eeg.shape),
            "eeg_mask": list(self.eeg_mask.shape),
            "eeg_lengths": list(self.eeg_lengths.shape),
            "context_words": (
                list(self.context_words.shape)
                if self.context_words is not None
                else None
            ),
            "word_mask": list(self.word_mask.shape),
            "word_lengths": list(self.word_lengths.shape),
            "word_keyword_indices": list(
                self.word_keyword_indices.shape
            ),
            "word_positions": list(self.word_positions.shape),
            "subject_indices": list(self.subject_indices.shape),
            "text_embedding_indices": list(
                self.text_embedding_indices.shape
            ),
        }
