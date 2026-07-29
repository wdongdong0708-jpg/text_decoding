from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class MasterKeywordIndex:
    keyword_ids: tuple[str, ...]
    keyword_to_index: dict[str, int]
    core_indices: frozenset[int]
    main_indices: frozenset[int]
    extended_indices: frozenset[int]
    master_indices: frozenset[int]
    mapping_sha256: str

    def index_or_minus_one(self, keyword_id: str) -> int:
        if not keyword_id:
            return -1
        try:
            return self.keyword_to_index[keyword_id]
        except KeyError as error:
            raise KeyError(
                f"Unknown non-empty Master keyword ID: {keyword_id!r}"
            ) from error

    def indices(self, keyword_ids: Iterable[str]) -> tuple[int, ...]:
        return tuple(self.index_or_minus_one(value) for value in keyword_ids)

    def to_metadata(self) -> dict[str, object]:
        return {
            "ordering_rule": "ascending frozen lexicon rank",
            "keyword_ids": list(self.keyword_ids),
            "mapping_sha256": self.mapping_sha256,
            "core_size": len(self.core_indices),
            "main_size": len(self.main_indices),
            "extended_size": len(self.extended_indices),
            "master_size": len(self.master_indices),
        }


def build_master_keyword_index(
    lexicon_path: str | Path,
) -> MasterKeywordIndex:
    path = Path(lexicon_path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Lexicon is empty: {path}")
    rows = sorted(rows, key=lambda row: int(row["rank"]))
    ranks = [int(row["rank"]) for row in rows]
    if ranks != list(range(1, len(rows) + 1)):
        raise ValueError("Frozen Master lexicon ranks must be contiguous")
    keyword_ids = tuple(row["keyword_id"] for row in rows)
    if any(not value for value in keyword_ids):
        raise ValueError("Master keyword IDs must be non-empty")
    if len(set(keyword_ids)) != len(keyword_ids):
        raise ValueError("Duplicate Master keyword ID")
    mapping = {
        keyword_id: index for index, keyword_id in enumerate(keyword_ids)
    }

    def included(field: str) -> frozenset[int]:
        return frozenset(
            index
            for index, row in enumerate(rows)
            if row[field] == "true"
        )

    core = included("include_core")
    main = included("include_main")
    extended = included("include_extended")
    master = frozenset(range(len(rows)))
    if not core <= main <= extended <= master:
        raise ValueError("Core/Main/Extended/Master tiers are not nested")
    canonical = json.dumps(
        keyword_ids,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return MasterKeywordIndex(
        keyword_ids=keyword_ids,
        keyword_to_index=mapping,
        core_indices=core,
        main_indices=main,
        extended_indices=extended,
        master_indices=master,
        mapping_sha256=sha256(canonical).hexdigest(),
    )
