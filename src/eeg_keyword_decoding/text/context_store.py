from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from eeg_keyword_decoding.data.protocol_assets import file_sha256

from .schema import (
    CONTEXT_WORD_STORE_SCHEMA_VERSION,
    ArrayDescriptor,
    ContextSentence,
    ContextWordStoreMetadata,
)


ARRAY_FILENAMES = {
    "context_vectors": "context_vectors.npy",
    "text_embedding_indices": "text_embedding_indices.npy",
    "sentence_offsets": "sentence_offsets.npy",
    "word_occurrence_ids": "word_occurrence_ids.npy",
    "word_positions": "word_positions.npy",
    "surface_forms": "surface_forms.npy",
    "char_spans": "char_spans.npy",
    "keyword_ids": "keyword_ids.npy",
}
METADATA_FILENAME = "metadata.json"


@dataclass(frozen=True)
class SentenceContextWords:
    text_embedding_idx: int
    vectors: np.ndarray
    word_occurrence_ids: np.ndarray
    word_positions: np.ndarray
    surface_forms: np.ndarray
    char_spans: np.ndarray
    keyword_ids: np.ndarray


def _unicode_array(values: Sequence[str]) -> np.ndarray:
    max_length = max((len(value) for value in values), default=1)
    return np.asarray(values, dtype=f"<U{max(max_length, 1)}")


def _sentence_index_arrays(
    sentences: Sequence[ContextSentence],
) -> dict[str, np.ndarray]:
    text_embedding_indices = np.asarray(
        [sentence.text_embedding_idx for sentence in sentences],
        dtype=np.int32,
    )
    if len(set(text_embedding_indices.tolist())) != len(sentences):
        raise ValueError("Duplicate text_embedding_idx in context sentences")

    offsets = np.zeros(len(sentences) + 1, dtype=np.int64)
    for index, sentence in enumerate(sentences):
        offsets[index + 1] = offsets[index] + sentence.word_count
    occurrences = [
        occurrence
        for sentence in sentences
        for occurrence in sentence.occurrences
    ]
    return {
        "text_embedding_indices": text_embedding_indices,
        "sentence_offsets": offsets,
        "word_occurrence_ids": _unicode_array(
            [item.word_occurrence_id for item in occurrences]
        ),
        "word_positions": np.asarray(
            [item.word_position for item in occurrences],
            dtype=np.int32,
        ),
        "surface_forms": _unicode_array(
            [item.surface_form for item in occurrences]
        ),
        "char_spans": np.asarray(
            [[item.char_start, item.char_end] for item in occurrences],
            dtype=np.int32,
        ),
        "keyword_ids": _unicode_array([item.keyword_id for item in occurrences]),
    }


def _save_array_atomic(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    temporary.replace(path)


def write_context_word_store(
    output_dir: str | Path,
    *,
    vectors: np.ndarray,
    sentences: Sequence[ContextSentence],
    metadata_fields: dict[str, Any],
    overwrite: bool = False,
) -> ContextWordStoreMetadata:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    known_paths = [
        output_root / filename for filename in ARRAY_FILENAMES.values()
    ] + [output_root / METADATA_FILENAME]
    existing = [path for path in known_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Context store already contains generated files: "
            + ", ".join(str(path) for path in existing)
        )

    index_arrays = _sentence_index_arrays(sentences)
    context_vectors = np.asarray(vectors)
    total_words = int(index_arrays["sentence_offsets"][-1])
    if context_vectors.shape[0] != total_words:
        raise ValueError(
            f"Vector row count {context_vectors.shape[0]} != {total_words}"
        )
    if context_vectors.ndim < 2:
        raise ValueError(
            "Context vectors must have a leading word axis and at least one "
            "feature axis"
        )
    arrays = {"context_vectors": context_vectors, **index_arrays}

    descriptors: dict[str, ArrayDescriptor] = {}
    for name, filename in ARRAY_FILENAMES.items():
        array = arrays[name]
        path = output_root / filename
        _save_array_atomic(path, array)
        descriptors[name] = ArrayDescriptor(
            filename=filename,
            dtype=str(array.dtype),
            shape=tuple(int(value) for value in array.shape),
            sha256=file_sha256(path),
            file_size_bytes=path.stat().st_size,
        )

    required_fields = {
        "model",
        "tokenizer",
        "extraction",
        "sources",
        "generation",
        "runtime",
    }
    missing = required_fields - set(metadata_fields)
    if missing:
        raise ValueError(f"Missing metadata fields: {sorted(missing)}")
    metadata = ContextWordStoreMetadata(
        schema_version=CONTEXT_WORD_STORE_SCHEMA_VERSION,
        arrays=descriptors,
        model=dict(metadata_fields["model"]),
        tokenizer=dict(metadata_fields["tokenizer"]),
        extraction=dict(metadata_fields["extraction"]),
        sources=dict(metadata_fields["sources"]),
        generation=dict(metadata_fields["generation"]),
        runtime=dict(metadata_fields["runtime"]),
    )
    metadata_path = output_root / METADATA_FILENAME
    temporary = metadata_path.with_name(f".{metadata_path.name}.tmp")
    temporary.write_text(
        json.dumps(
            metadata.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(metadata_path)
    return metadata


class ContextWordStore:
    def __init__(
        self,
        root: str | Path,
        *,
        mmap_mode: str | None = "r",
        verify_hashes: bool = False,
    ) -> None:
        self.root = Path(root)
        self.metadata = ContextWordStoreMetadata.read(
            self.root / METADATA_FILENAME
        )
        self._arrays: dict[str, np.ndarray] = {}
        for name, descriptor in self.metadata.arrays.items():
            path = self.root / descriptor.filename
            if not path.is_file():
                raise FileNotFoundError(path)
            if verify_hashes and file_sha256(path) != descriptor.sha256:
                raise ValueError(f"SHA256 mismatch for {path}")
            array = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
            if str(array.dtype) != descriptor.dtype:
                raise ValueError(f"dtype mismatch for {path}")
            if tuple(array.shape) != descriptor.shape:
                raise ValueError(f"shape mismatch for {path}")
            if path.stat().st_size != descriptor.file_size_bytes:
                raise ValueError(f"file size mismatch for {path}")
            self._arrays[name] = array

        missing = set(ARRAY_FILENAMES) - set(self._arrays)
        if missing:
            raise ValueError(f"Context store is missing arrays: {sorted(missing)}")
        self._validate_contract()
        indices = self.text_embedding_indices.tolist()
        self._sentence_row_by_idx = {
            int(text_embedding_idx): row
            for row, text_embedding_idx in enumerate(indices)
        }

    def _validate_contract(self) -> None:
        sentence_count = len(self.text_embedding_indices)
        if self.sentence_offsets.shape != (sentence_count + 1,):
            raise ValueError("sentence_offsets must have sentences + 1 entries")
        if int(self.sentence_offsets[0]) != 0:
            raise ValueError("sentence_offsets must start at zero")
        if np.any(np.diff(self.sentence_offsets) <= 0):
            raise ValueError("Every cached sentence must contain at least one word")
        total_words = int(self.sentence_offsets[-1])
        word_arrays = (
            self.context_vectors,
            self.word_occurrence_ids,
            self.word_positions,
            self.surface_forms,
            self.char_spans,
            self.keyword_ids,
        )
        if any(array.shape[0] != total_words for array in word_arrays):
            raise ValueError("Word-level context store arrays have unequal rows")
        if self.char_spans.shape != (total_words, 2):
            raise ValueError("char_spans must have shape [total_words, 2]")
        if len(set(self.text_embedding_indices.tolist())) != sentence_count:
            raise ValueError("text_embedding_indices must be unique")

    @property
    def context_vectors(self) -> np.ndarray:
        return self._arrays["context_vectors"]

    @property
    def text_embedding_indices(self) -> np.ndarray:
        return self._arrays["text_embedding_indices"]

    @property
    def sentence_offsets(self) -> np.ndarray:
        return self._arrays["sentence_offsets"]

    @property
    def word_occurrence_ids(self) -> np.ndarray:
        return self._arrays["word_occurrence_ids"]

    @property
    def word_positions(self) -> np.ndarray:
        return self._arrays["word_positions"]

    @property
    def surface_forms(self) -> np.ndarray:
        return self._arrays["surface_forms"]

    @property
    def char_spans(self) -> np.ndarray:
        return self._arrays["char_spans"]

    @property
    def keyword_ids(self) -> np.ndarray:
        return self._arrays["keyword_ids"]

    def get_sentence(
        self,
        text_embedding_idx: int,
        *,
        vector_dtype: str | np.dtype[Any] | None = None,
    ) -> SentenceContextWords:
        try:
            sentence_row = self._sentence_row_by_idx[int(text_embedding_idx)]
        except KeyError as error:
            raise KeyError(
                f"Unknown text_embedding_idx: {text_embedding_idx}"
            ) from error
        start = int(self.sentence_offsets[sentence_row])
        stop = int(self.sentence_offsets[sentence_row + 1])
        vectors = self.context_vectors[start:stop]
        if vector_dtype is not None:
            vectors = vectors.astype(vector_dtype, copy=False)
        return SentenceContextWords(
            text_embedding_idx=int(text_embedding_idx),
            vectors=vectors,
            word_occurrence_ids=self.word_occurrence_ids[start:stop],
            word_positions=self.word_positions[start:stop],
            surface_forms=self.surface_forms[start:stop],
            char_spans=self.char_spans[start:stop],
            keyword_ids=self.keyword_ids[start:stop],
        )

    def audit_against(self, sentences: Sequence[ContextSentence]) -> None:
        expected = _sentence_index_arrays(sentences)
        for name, array in expected.items():
            np.testing.assert_array_equal(self._arrays[name], array)
        expected_words = sum(sentence.word_count for sentence in sentences)
        if self.context_vectors.shape[0] != expected_words:
            raise AssertionError(
                f"Expected {expected_words} vector rows, got "
                f"{self.context_vectors.shape[0]}"
            )
