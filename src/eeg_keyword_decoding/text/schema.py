from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eeg_keyword_decoding.data.word_occurrences import normalize_text


CONTEXT_WORD_STORE_SCHEMA_VERSION = "context_word_store_v1"


@dataclass(frozen=True)
class ContextWordOccurrence:
    word_occurrence_id: str
    text_embedding_idx: int
    word_position: int
    surface_form: str
    char_start: int
    char_end: int
    keyword_id: str


@dataclass(frozen=True)
class ContextSentence:
    text_embedding_idx: int
    text: str
    occurrences: tuple[ContextWordOccurrence, ...]

    @property
    def word_count(self) -> int:
        return len(self.occurrences)


@dataclass(frozen=True)
class ArrayDescriptor:
    filename: str
    dtype: str
    shape: tuple[int, ...]
    sha256: str
    file_size_bytes: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ArrayDescriptor:
        return cls(
            filename=str(value["filename"]),
            dtype=str(value["dtype"]),
            shape=tuple(int(item) for item in value["shape"]),
            sha256=str(value["sha256"]),
            file_size_bytes=int(value["file_size_bytes"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "sha256": self.sha256,
            "file_size_bytes": self.file_size_bytes,
        }


@dataclass(frozen=True)
class ContextWordStoreMetadata:
    schema_version: str
    arrays: dict[str, ArrayDescriptor]
    model: dict[str, Any]
    tokenizer: dict[str, Any]
    extraction: dict[str, Any]
    sources: dict[str, Any]
    generation: dict[str, Any]
    runtime: dict[str, Any]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ContextWordStoreMetadata:
        schema_version = str(value["schema_version"])
        if schema_version != CONTEXT_WORD_STORE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported context store schema: {schema_version!r}"
            )
        arrays = {
            name: ArrayDescriptor.from_dict(descriptor)
            for name, descriptor in dict(value["arrays"]).items()
        }
        return cls(
            schema_version=schema_version,
            arrays=arrays,
            model=dict(value["model"]),
            tokenizer=dict(value["tokenizer"]),
            extraction=dict(value["extraction"]),
            sources=dict(value["sources"]),
            generation=dict(value["generation"]),
            runtime=dict(value["runtime"]),
        )

    @classmethod
    def read(cls, path: str | Path) -> ContextWordStoreMetadata:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "arrays": {
                name: descriptor.to_dict()
                for name, descriptor in sorted(self.arrays.items())
            },
            "model": self.model,
            "tokenizer": self.tokenizer,
            "extraction": self.extraction,
            "sources": self.sources,
            "generation": self.generation,
            "runtime": self.runtime,
        }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def load_context_sentences(
    sentence_labels_path: str | Path,
    word_occurrences_path: str | Path,
) -> list[ContextSentence]:
    """Load full sentences in the exact frozen occurrence-table row order."""

    label_rows = _read_csv(Path(sentence_labels_path))
    occurrence_rows = _read_csv(Path(word_occurrences_path))
    valid_labels = [
        row
        for row in label_rows
        if row["is_chapter_heading"] != "true" and int(row["token_count"]) > 0
    ]
    label_by_idx = {
        int(row["text_embedding_idx"]): row for row in valid_labels
    }

    grouped: dict[int, list[ContextWordOccurrence]] = {}
    sentence_order: list[int] = []
    previous_idx: int | None = None
    closed_indices: set[int] = set()
    for row in occurrence_rows:
        text_embedding_idx = int(row["text_embedding_idx"])
        if text_embedding_idx not in label_by_idx:
            raise ValueError(
                "Occurrence references an excluded or unknown sentence: "
                f"{text_embedding_idx}"
            )
        if text_embedding_idx != previous_idx:
            if text_embedding_idx in closed_indices:
                raise ValueError(
                    "Occurrence rows for a sentence must be contiguous: "
                    f"{text_embedding_idx}"
                )
            if previous_idx is not None:
                closed_indices.add(previous_idx)
            sentence_order.append(text_embedding_idx)
            grouped[text_embedding_idx] = []
            previous_idx = text_embedding_idx

        grouped[text_embedding_idx].append(
            ContextWordOccurrence(
                word_occurrence_id=row["word_occurrence_id"],
                text_embedding_idx=text_embedding_idx,
                word_position=int(row["word_position"]),
                surface_form=row["surface_form"],
                char_start=int(row["char_start"]),
                char_end=int(row["char_end"]),
                keyword_id=row["keyword_id"],
            )
        )

    expected_order = [
        int(row["text_embedding_idx"]) for row in valid_labels
    ]
    if sentence_order != expected_order:
        raise ValueError(
            "Frozen occurrence sentence order does not match sentence labels"
        )

    sentences: list[ContextSentence] = []
    for text_embedding_idx in sentence_order:
        label = label_by_idx[text_embedding_idx]
        text = normalize_text(label["text"])
        occurrences = tuple(grouped[text_embedding_idx])
        expected_positions = list(range(len(occurrences)))
        actual_positions = [item.word_position for item in occurrences]
        if actual_positions != expected_positions:
            raise ValueError(
                f"Non-contiguous word positions for {text_embedding_idx}: "
                f"{actual_positions}"
            )
        if len(occurrences) != int(label["token_count"]):
            raise ValueError(
                f"token_count mismatch for {text_embedding_idx}: "
                f"{len(occurrences)} != {label['token_count']}"
            )
        for occurrence in occurrences:
            if not (
                0 <= occurrence.char_start < occurrence.char_end <= len(text)
            ):
                raise ValueError(
                    f"Invalid character span for {occurrence.word_occurrence_id}"
                )
            observed = text[occurrence.char_start : occurrence.char_end]
            if observed != occurrence.surface_form:
                raise ValueError(
                    f"Surface/span mismatch for {occurrence.word_occurrence_id}: "
                    f"{observed!r} != {occurrence.surface_form!r}"
                )
        sentences.append(
            ContextSentence(
                text_embedding_idx=text_embedding_idx,
                text=text,
                occurrences=occurrences,
            )
        )

    if sum(sentence.word_count for sentence in sentences) != len(
        occurrence_rows
    ):
        raise ValueError("Occurrence rows were lost while loading sentences")
    return sentences
