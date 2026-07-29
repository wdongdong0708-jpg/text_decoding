from __future__ import annotations

import csv
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch.utils.data import Dataset

from eeg_keyword_decoding.io import BrainVisionReader
from eeg_keyword_decoding.text import (
    ContextSentence,
    ContextWordStore,
    load_context_sentences,
)

from .eeg_manifest import EEGManifestRecord
from .keyword_index import MasterKeywordIndex
from .protocol_assets import file_sha256
from .sample import ContextEEGSample
from .split_index import SplitRole, SplitViewIndex
from .subject_index import SubjectIndex


class ContextTargetAccessError(PermissionError):
    """Raised when validation/test code requests contextual text targets."""


class EEGWindowReader(Protocol):
    n_channels: int
    sfreq: float

    def read_window(
        self,
        start_sample: int,
        stop_sample: int,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class ContextCacheFingerprint:
    backend: str
    metadata_sha256: str
    context_vectors_sha256: str
    stored_dtype: str
    feature_tail_shape: tuple[int, ...]
    model_revision: str
    tokenizer_revision: str


def _load_keyword_truth(
    sentence_labels_path: Path,
    *,
    valid_indices: frozenset[int],
    sentence_by_idx: dict[int, ContextSentence],
    keyword_index: MasterKeywordIndex,
) -> dict[int, tuple[int, ...]]:
    with sentence_labels_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    label_by_idx = {
        int(row["text_embedding_idx"]): row
        for row in rows
        if int(row["text_embedding_idx"]) in valid_indices
    }
    if set(label_by_idx) != set(valid_indices):
        raise ValueError(
            "Sentence keyword labels do not cover the frozen valid split"
        )

    truth: dict[int, tuple[int, ...]] = {}
    for text_embedding_idx in sorted(valid_indices):
        sentence = sentence_by_idx[text_embedding_idx]
        occurrence_keyword_ids = {
            occurrence.keyword_id
            for occurrence in sentence.occurrences
            if occurrence.keyword_id
        }
        declared_keyword_ids = {
            value
            for value in label_by_idx[text_embedding_idx][
                "present_keyword_ids"
            ].split("|")
            if value
        }
        if occurrence_keyword_ids != declared_keyword_ids:
            raise ValueError(
                "Occurrence-derived keyword truth disagrees with frozen "
                f"sentence labels for {text_embedding_idx}"
            )
        truth[text_embedding_idx] = tuple(
            sorted(
                keyword_index.index_or_minus_one(value)
                for value in declared_keyword_ids
            )
        )
    return truth


class ContextEEGDataset(Dataset[ContextEEGSample]):
    """One subject EEG view of one frozen sentence occurrence per sample."""

    def __init__(
        self,
        *,
        split_index: SplitViewIndex,
        outer_fold: int,
        role: SplitRole,
        subject_index: SubjectIndex,
        keyword_index: MasterKeywordIndex,
        sentence_labels_path: str | Path,
        word_occurrences_path: str | Path,
        text_backend: str | None = None,
        context_store_root: str | Path | None = None,
        include_context_targets: bool | None = None,
        expected_cache_metadata_sha256: str | None = None,
        expected_context_vectors_sha256: str | None = None,
        reader_factory: Callable[[str | Path], EEGWindowReader] = (
            BrainVisionReader
        ),
        expected_eeg_channels: int | None = None,
    ) -> None:
        if include_context_targets is None:
            include_context_targets = role == "train"
        if role != "train" and include_context_targets:
            raise ContextTargetAccessError(
                "Context targets are available only for inner-train; "
                f"role={role!r} requested include_context_targets=True"
            )
        if role != "train" and context_store_root is not None:
            raise ContextTargetAccessError(
                "Validation/test Dataset must not be wired to a context "
                "target store"
            )
        if include_context_targets and context_store_root is None:
            raise ValueError(
                "inner-train context targets require context_store_root"
            )
        if include_context_targets and not text_backend:
            raise ValueError(
                "inner-train context targets require a backend name"
            )
        if expected_eeg_channels is not None and expected_eeg_channels <= 0:
            raise ValueError("expected_eeg_channels must be positive")

        self.split_index = split_index
        self.outer_fold = int(outer_fold)
        self.role = role
        self.records = split_index.records_for(self.outer_fold, role)
        if not self.records:
            raise ValueError(
                f"No EEG views for outer_fold={outer_fold}, role={role}"
            )
        self.subject_index = subject_index
        self.keyword_index = keyword_index
        self.text_backend = text_backend
        self.include_context_targets = bool(include_context_targets)
        self.context_store_root = (
            Path(context_store_root).resolve()
            if context_store_root is not None
            else None
        )
        self.reader_factory = reader_factory
        self.expected_eeg_channels = expected_eeg_channels
        self._observed_eeg_channels: int | None = None
        self._resource_pid = os.getpid()
        self._reader_cache: dict[Path, EEGWindowReader] = {}
        self._context_store: ContextWordStore | None = None

        self.sentences = tuple(
            load_context_sentences(
                sentence_labels_path,
                word_occurrences_path,
            )
        )
        self._sentence_by_idx = {
            sentence.text_embedding_idx: sentence
            for sentence in self.sentences
        }
        if set(self._sentence_by_idx) != set(
            split_index.valid_text_embedding_indices
        ):
            raise ValueError(
                "Frozen word occurrence sentences do not match fold sentences"
            )
        for assignment in split_index.assignments.values():
            observed_count = self._sentence_by_idx[
                assignment.text_embedding_idx
            ].word_count
            if observed_count != assignment.token_count:
                raise ValueError(
                    "Fold token_count disagrees with frozen occurrences for "
                    f"{assignment.text_embedding_idx}"
                )
        self._present_keyword_indices = _load_keyword_truth(
            Path(sentence_labels_path),
            valid_indices=split_index.valid_text_embedding_indices,
            sentence_by_idx=self._sentence_by_idx,
            keyword_index=keyword_index,
        )

        self.context_cache_fingerprint: ContextCacheFingerprint | None = None
        if self.include_context_targets:
            assert self.context_store_root is not None
            store = ContextWordStore(
                self.context_store_root,
                verify_hashes=True,
            )
            store.audit_against(self.sentences)
            metadata_path = self.context_store_root / "metadata.json"
            metadata_sha256 = file_sha256(metadata_path)
            vector_descriptor = store.metadata.arrays["context_vectors"]
            if (
                expected_cache_metadata_sha256 is not None
                and metadata_sha256 != expected_cache_metadata_sha256
            ):
                raise ValueError(
                    "Context cache metadata SHA256 mismatch: "
                    f"{metadata_sha256} != "
                    f"{expected_cache_metadata_sha256}"
                )
            if (
                expected_context_vectors_sha256 is not None
                and vector_descriptor.sha256
                != expected_context_vectors_sha256
            ):
                raise ValueError(
                    "Context vector SHA256 mismatch: "
                    f"{vector_descriptor.sha256} != "
                    f"{expected_context_vectors_sha256}"
                )
            self.context_cache_fingerprint = ContextCacheFingerprint(
                backend=str(text_backend),
                metadata_sha256=metadata_sha256,
                context_vectors_sha256=vector_descriptor.sha256,
                stored_dtype=vector_descriptor.dtype,
                feature_tail_shape=tuple(vector_descriptor.shape[1:]),
                model_revision=str(
                    store.metadata.model["resolved_revision"]
                ),
                tokenizer_revision=str(
                    store.metadata.tokenizer["resolved_revision"]
                ),
            )
            self._context_store = store

    def __len__(self) -> int:
        return len(self.records)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_reader_cache"] = {}
        state["_context_store"] = None
        state["_observed_eeg_channels"] = None
        state["_resource_pid"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._resource_pid = os.getpid()
        self._reader_cache = {}
        self._context_store = None
        self._observed_eeg_channels = None

    def _ensure_process_resources(self) -> None:
        current_pid = os.getpid()
        if self._resource_pid != current_pid:
            self._resource_pid = current_pid
            self._reader_cache = {}
            self._context_store = None
            self._observed_eeg_channels = None

    def _reader_for(self, record: EEGManifestRecord) -> EEGWindowReader:
        self._ensure_process_resources()
        path = record.eeg_vhdr_path.resolve()
        reader = self._reader_cache.get(path)
        if reader is None:
            reader = self.reader_factory(path)
            self._reader_cache[path] = reader
        return reader

    def _get_context_store(self) -> ContextWordStore:
        if not self.include_context_targets:
            raise ContextTargetAccessError(
                f"Context targets are forbidden for role={self.role!r}"
            )
        self._ensure_process_resources()
        if self._context_store is None:
            assert self.context_store_root is not None
            self._context_store = ContextWordStore(
                self.context_store_root,
                verify_hashes=False,
            )
        return self._context_store

    def _read_eeg(self, record: EEGManifestRecord) -> torch.Tensor:
        reader = self._reader_for(record)
        reader_sfreq = getattr(reader, "sfreq", record.sfreq)
        if not np.isclose(float(reader_sfreq), record.sfreq):
            raise ValueError(
                f"Sampling frequency mismatch for {record.eeg_view_id}: "
                f"{reader_sfreq} != {record.sfreq}"
            )
        value = reader.read_window(
            record.start_sample,
            record.stop_sample,
        )
        eeg = np.array(value, dtype=np.float32, copy=True, order="C")
        if eeg.ndim != 2:
            raise ValueError(
                f"EEG reader must return [channel, time], got {eeg.shape}"
            )
        channels, length = eeg.shape
        if channels <= 0 or length <= 0:
            raise ValueError(
                f"Empty EEG window for {record.eeg_view_id}: {eeg.shape}"
            )
        if length != record.n_samples:
            raise ValueError(
                f"EEG length mismatch for {record.eeg_view_id}: "
                f"{length} != {record.n_samples}"
            )
        expected = self.expected_eeg_channels
        if expected is None:
            if self._observed_eeg_channels is None:
                self._observed_eeg_channels = channels
            expected = self._observed_eeg_channels
        if channels != expected:
            raise ValueError(
                f"EEG channel mismatch for {record.eeg_view_id}: "
                f"{channels} != {expected}"
            )
        return torch.from_numpy(eeg)

    def __getitem__(self, index: int) -> ContextEEGSample:
        record = self.records[index]
        assignment = self.split_index.assignment(
            self.outer_fold,
            record.text_embedding_idx,
        )
        if assignment.role != self.role:
            raise RuntimeError("Split index returned a cross-role EEG view")
        sentence = self._sentence_by_idx[record.text_embedding_idx]
        eeg = self._read_eeg(record)
        keyword_ids = tuple(
            occurrence.keyword_id for occurrence in sentence.occurrences
        )
        keyword_indices = torch.tensor(
            self.keyword_index.indices(keyword_ids),
            dtype=torch.int64,
        )
        context_words: torch.Tensor | None = None
        if self.include_context_targets:
            cached = self._get_context_store().get_sentence(
                record.text_embedding_idx,
                vector_dtype=np.float32,
            )
            if cached.vectors.shape[0] != sentence.word_count:
                raise ValueError(
                    "Context cache word count changed after initialization for "
                    f"{record.text_embedding_idx}"
                )
            context_words = torch.from_numpy(
                np.array(
                    cached.vectors,
                    dtype=np.float32,
                    copy=True,
                    order="C",
                )
            )

        fingerprint = self.context_cache_fingerprint
        return ContextEEGSample(
            eeg_view_id=record.eeg_view_id,
            subject=record.subject,
            subject_index=self.subject_index.index(record.subject),
            session=record.session,
            task=record.task,
            run=record.run,
            local_row_idx=record.local_row_idx,
            global_row_idx=record.global_row_idx,
            text_embedding_idx=record.text_embedding_idx,
            sentence_group_id=assignment.sentence_group_id,
            outer_fold=self.outer_fold,
            role=self.role,
            sfreq=record.sfreq,
            start_sample=record.start_sample,
            stop_sample=record.stop_sample,
            eeg=eeg,
            eeg_length=record.n_samples,
            word_occurrence_ids=tuple(
                occurrence.word_occurrence_id
                for occurrence in sentence.occurrences
            ),
            word_positions=torch.tensor(
                [
                    occurrence.word_position
                    for occurrence in sentence.occurrences
                ],
                dtype=torch.int64,
            ),
            word_surface_forms=tuple(
                occurrence.surface_form
                for occurrence in sentence.occurrences
            ),
            word_char_spans=torch.tensor(
                [
                    [occurrence.char_start, occurrence.char_end]
                    for occurrence in sentence.occurrences
                ],
                dtype=torch.int64,
            ),
            word_keyword_ids=keyword_ids,
            word_keyword_indices=keyword_indices,
            present_keyword_indices=torch.tensor(
                self._present_keyword_indices[record.text_embedding_idx],
                dtype=torch.int64,
            ),
            context_words=context_words,
            context_backend=self.text_backend,
            context_cache_metadata_sha256=(
                fingerprint.metadata_sha256
                if fingerprint is not None
                else None
            ),
            context_vectors_sha256=(
                fingerprint.context_vectors_sha256
                if fingerprint is not None
                else None
            ),
        )
