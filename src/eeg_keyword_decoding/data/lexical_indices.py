from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

from .protocol_assets import file_sha256
from .split_index import SplitViewIndex


LEXICAL_IDENTITY_SCHEMA_VERSION = "littleprince_lexical_identity_v1"


@dataclass(frozen=True, order=True)
class ContextTokenKey:
    sentence_group_id: str
    word_position: int
    surface_form: str

    @property
    def stable_id(self) -> str:
        payload = json.dumps(
            [
                self.sentence_group_id,
                self.word_position,
                self.surface_form,
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return "ctg:" + sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LexicalIdentityIndex:
    """Stable identities for occurrence-, context-, and lexical-level words."""

    surface_types: tuple[str, ...]
    surface_to_index: dict[str, int]
    context_token_keys: tuple[ContextTokenKey, ...]
    context_token_stable_ids: tuple[str, ...]
    occurrence_to_surface_index: dict[str, int]
    occurrence_to_context_token_group_index: dict[str, int]
    sentence_group_ids: tuple[str, ...]
    sentence_group_to_index: dict[str, int]
    source_word_occurrences_sha256: str
    source_fold_sha256: str
    mapping_sha256: str

    @property
    def occurrence_count(self) -> int:
        return len(self.occurrence_to_surface_index)

    @property
    def surface_type_count(self) -> int:
        return len(self.surface_types)

    @property
    def context_token_group_count(self) -> int:
        return len(self.context_token_keys)

    def surface_indices(
        self,
        word_occurrence_ids: Iterable[str],
    ) -> tuple[int, ...]:
        return tuple(
            self.occurrence_to_surface_index[occurrence_id]
            for occurrence_id in word_occurrence_ids
        )

    def context_token_group_indices(
        self,
        word_occurrence_ids: Iterable[str],
    ) -> tuple[int, ...]:
        return tuple(
            self.occurrence_to_context_token_group_index[occurrence_id]
            for occurrence_id in word_occurrence_ids
        )

    def sentence_group_index(self, sentence_group_id: str) -> int:
        try:
            return self.sentence_group_to_index[sentence_group_id]
        except KeyError as error:
            raise KeyError(
                f"Unknown frozen sentence_group_id: {sentence_group_id!r}"
            ) from error

    def to_metadata(self, *, include_mappings: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": LEXICAL_IDENTITY_SCHEMA_VERSION,
            "ordering_rules": {
                "surface_type": "ascending Unicode code-point order",
                "context_token_group": (
                    "ascending tuple(sentence_group_id, word_position, "
                    "surface_form)"
                ),
                "sentence_group": "ascending sentence_group_id",
            },
            "source_word_occurrences_sha256": (
                self.source_word_occurrences_sha256
            ),
            "source_fold_sha256": self.source_fold_sha256,
            "mapping_sha256": self.mapping_sha256,
            "word_occurrence_count": self.occurrence_count,
            "surface_type_count": self.surface_type_count,
            "context_token_group_count": self.context_token_group_count,
            "sentence_group_count": len(self.sentence_group_ids),
        }
        if include_mappings:
            value.update(
                {
                    "surface_types": list(self.surface_types),
                    "context_token_groups": [
                        {
                            "stable_id": stable_id,
                            "sentence_group_id": key.sentence_group_id,
                            "word_position": key.word_position,
                            "surface_form": key.surface_form,
                        }
                        for key, stable_id in zip(
                            self.context_token_keys,
                            self.context_token_stable_ids,
                            strict=True,
                        )
                    ],
                    "sentence_group_ids": list(self.sentence_group_ids),
                    "occurrence_to_surface_index": dict(
                        sorted(self.occurrence_to_surface_index.items())
                    ),
                    "occurrence_to_context_token_group_index": dict(
                        sorted(
                            self.occurrence_to_context_token_group_index.items()
                        )
                    ),
                }
            )
        return value

    def write(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                self.to_metadata(include_mappings=True),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


def _read_occurrence_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Word occurrence CSV is empty: {path}")
    return rows


def _canonical_mapping_sha256(
    *,
    source_word_occurrences_sha256: str,
    source_fold_sha256: str,
    surface_types: tuple[str, ...],
    context_token_keys: tuple[ContextTokenKey, ...],
    sentence_group_ids: tuple[str, ...],
    occurrence_to_surface_index: dict[str, int],
    occurrence_to_context_token_group_index: dict[str, int],
) -> str:
    payload = {
        "schema_version": LEXICAL_IDENTITY_SCHEMA_VERSION,
        "source_word_occurrences_sha256": source_word_occurrences_sha256,
        "source_fold_sha256": source_fold_sha256,
        "surface_types": list(surface_types),
        "context_token_keys": [
            [
                key.sentence_group_id,
                key.word_position,
                key.surface_form,
            ]
            for key in context_token_keys
        ],
        "sentence_group_ids": list(sentence_group_ids),
        "occurrence_to_surface_index": sorted(
            occurrence_to_surface_index.items()
        ),
        "occurrence_to_context_token_group_index": sorted(
            occurrence_to_context_token_group_index.items()
        ),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def build_lexical_identity_index(
    word_occurrences_path: str | Path,
    split_index: SplitViewIndex,
) -> LexicalIdentityIndex:
    occurrence_file = Path(word_occurrences_path)
    rows = _read_occurrence_rows(occurrence_file)

    group_by_sentence: dict[int, str] = {}
    for assignment in split_index.assignments.values():
        previous = group_by_sentence.setdefault(
            assignment.text_embedding_idx,
            assignment.sentence_group_id,
        )
        if previous != assignment.sentence_group_id:
            raise ValueError(
                "sentence_group_id changes across folds for "
                f"text_embedding_idx={assignment.text_embedding_idx}"
            )

    occurrence_ids = [row["word_occurrence_id"] for row in rows]
    if len(occurrence_ids) != len(set(occurrence_ids)):
        raise ValueError("Duplicate word_occurrence_id in frozen table")
    referenced_sentences = {int(row["text_embedding_idx"]) for row in rows}
    if referenced_sentences != set(split_index.valid_text_embedding_indices):
        raise ValueError(
            "Word occurrence sentences do not match the frozen fold index"
        )

    surface_types = tuple(sorted({row["surface_form"] for row in rows}))
    surface_to_index = {
        surface: index for index, surface in enumerate(surface_types)
    }
    keys_by_occurrence: dict[str, ContextTokenKey] = {}
    for row in rows:
        text_embedding_idx = int(row["text_embedding_idx"])
        key = ContextTokenKey(
            sentence_group_id=group_by_sentence[text_embedding_idx],
            word_position=int(row["word_position"]),
            surface_form=row["surface_form"],
        )
        keys_by_occurrence[row["word_occurrence_id"]] = key

    context_token_keys = tuple(sorted(set(keys_by_occurrence.values())))
    key_to_index = {
        key: index for index, key in enumerate(context_token_keys)
    }
    stable_ids = tuple(key.stable_id for key in context_token_keys)
    if len(stable_ids) != len(set(stable_ids)):
        raise RuntimeError("Context-token stable ID collision")

    occurrence_to_surface_index = {
        row["word_occurrence_id"]: surface_to_index[row["surface_form"]]
        for row in rows
    }
    occurrence_to_context_token_group_index = {
        occurrence_id: key_to_index[key]
        for occurrence_id, key in keys_by_occurrence.items()
    }
    sentence_group_ids = tuple(sorted(set(group_by_sentence.values())))
    sentence_group_to_index = {
        group_id: index
        for index, group_id in enumerate(sentence_group_ids)
    }
    occurrence_sha256 = file_sha256(occurrence_file)
    mapping_sha256 = _canonical_mapping_sha256(
        source_word_occurrences_sha256=occurrence_sha256,
        source_fold_sha256=split_index.fold_source_sha256,
        surface_types=surface_types,
        context_token_keys=context_token_keys,
        sentence_group_ids=sentence_group_ids,
        occurrence_to_surface_index=occurrence_to_surface_index,
        occurrence_to_context_token_group_index=(
            occurrence_to_context_token_group_index
        ),
    )
    return LexicalIdentityIndex(
        surface_types=surface_types,
        surface_to_index=surface_to_index,
        context_token_keys=context_token_keys,
        context_token_stable_ids=stable_ids,
        occurrence_to_surface_index=occurrence_to_surface_index,
        occurrence_to_context_token_group_index=(
            occurrence_to_context_token_group_index
        ),
        sentence_group_ids=sentence_group_ids,
        sentence_group_to_index=sentence_group_to_index,
        source_word_occurrences_sha256=occurrence_sha256,
        source_fold_sha256=split_index.fold_source_sha256,
        mapping_sha256=mapping_sha256,
    )
