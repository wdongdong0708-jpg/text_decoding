from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .provenance import file_sha256
from .schema import PROTOTYPE_BANK_SCHEMA_VERSION, PrototypeBank


def _save_array(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temporary, path)


def save_prototype_bank(
    bank: PrototypeBank,
    directory: str | Path,
) -> dict[str, Any]:
    bank = bank.detached()
    bank.validate()
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    arrays = {
        "vectors": bank.vectors.cpu().numpy().astype(np.float32, copy=False),
        "available_mask": bank.available_mask.cpu().numpy().astype(
            np.bool_,
            copy=False,
        ),
        "train_sentence_df": bank.train_sentence_df.cpu().numpy().astype(
            np.int64,
            copy=False,
        ),
        "train_group_df": bank.train_group_df.cpu().numpy().astype(
            np.int64,
            copy=False,
        ),
    }
    descriptors: dict[str, Any] = {}
    for name, value in arrays.items():
        path = root / f"{name}.npy"
        _save_array(path, value)
        descriptors[name] = {
            "filename": path.name,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": file_sha256(path),
            "file_size_bytes": path.stat().st_size,
        }

    metadata = {
        "schema_version": PROTOTYPE_BANK_SCHEMA_VERSION,
        "outer_fold": bank.outer_fold,
        "text_backend": bank.text_backend,
        "keyword_ids": list(bank.keyword_ids),
        "available_count": bank.available_count,
        "projector_state_hash": bank.projector_state_hash,
        "source_cache_hash": bank.source_cache_hash,
        "source_cache_metadata_hash": bank.source_cache_metadata_hash,
        "fold_hash": bank.fold_hash,
        "eligibility_hash": bank.eligibility_hash,
        "lexical_mapping_hash": bank.lexical_mapping_hash,
        "arrays": descriptors,
        "builder": bank.metadata,
    }
    metadata_path = root / "metadata.json"
    temporary = metadata_path.with_name(metadata_path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, metadata_path)
    return {
        "directory": str(root.resolve()),
        "metadata_sha256": file_sha256(metadata_path),
        "arrays": descriptors,
        "available_count": bank.available_count,
    }


def _require_equal(name: str, observed: object, expected: object) -> None:
    if observed != expected:
        raise ValueError(
            f"Prototype bank {name} mismatch: {observed!r} != {expected!r}"
        )


def load_prototype_bank(
    directory: str | Path,
    *,
    expected_outer_fold: int,
    expected_text_backend: str,
    expected_projector_state_hash: str,
    expected_source_cache_hash: str,
    expected_fold_hash: str,
    expected_keyword_ids: tuple[str, ...],
    expected_eligibility_hash: str | None = None,
    expected_lexical_mapping_hash: str | None = None,
) -> PrototypeBank:
    root = Path(directory)
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    _require_equal(
        "schema_version",
        metadata.get("schema_version"),
        PROTOTYPE_BANK_SCHEMA_VERSION,
    )
    _require_equal("outer_fold", int(metadata["outer_fold"]), expected_outer_fold)
    _require_equal(
        "text_backend",
        str(metadata["text_backend"]),
        expected_text_backend,
    )
    _require_equal(
        "projector_state_hash",
        str(metadata["projector_state_hash"]),
        expected_projector_state_hash,
    )
    _require_equal(
        "source_cache_hash",
        str(metadata["source_cache_hash"]),
        expected_source_cache_hash,
    )
    _require_equal(
        "fold_hash",
        str(metadata["fold_hash"]),
        expected_fold_hash,
    )
    _require_equal(
        "keyword_ids",
        tuple(str(item) for item in metadata["keyword_ids"]),
        expected_keyword_ids,
    )
    if expected_eligibility_hash is not None:
        _require_equal(
            "eligibility_hash",
            str(metadata["eligibility_hash"]),
            expected_eligibility_hash,
        )
    if expected_lexical_mapping_hash is not None:
        _require_equal(
            "lexical_mapping_hash",
            str(metadata["lexical_mapping_hash"]),
            expected_lexical_mapping_hash,
        )

    loaded: dict[str, torch.Tensor] = {}
    for name, descriptor in dict(metadata["arrays"]).items():
        path = root / str(descriptor["filename"])
        _require_equal(
            f"{name}.sha256",
            file_sha256(path),
            str(descriptor["sha256"]),
        )
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        _require_equal(
            f"{name}.shape",
            tuple(value.shape),
            tuple(int(item) for item in descriptor["shape"]),
        )
        _require_equal(f"{name}.dtype", str(value.dtype), str(descriptor["dtype"]))
        loaded[name] = torch.from_numpy(np.array(value, copy=True))

    bank = PrototypeBank(
        vectors=loaded["vectors"].to(torch.float32),
        available_mask=loaded["available_mask"].to(torch.bool),
        keyword_ids=tuple(str(item) for item in metadata["keyword_ids"]),
        train_sentence_df=loaded["train_sentence_df"].to(torch.int64),
        train_group_df=loaded["train_group_df"].to(torch.int64),
        outer_fold=int(metadata["outer_fold"]),
        text_backend=str(metadata["text_backend"]),
        projector_state_hash=str(metadata["projector_state_hash"]),
        source_cache_hash=str(metadata["source_cache_hash"]),
        source_cache_metadata_hash=str(
            metadata["source_cache_metadata_hash"]
        ),
        fold_hash=str(metadata["fold_hash"]),
        eligibility_hash=str(metadata["eligibility_hash"]),
        lexical_mapping_hash=str(metadata["lexical_mapping_hash"]),
        metadata=dict(metadata["builder"]),
    )
    bank.validate()
    return replace(bank, vectors=bank.vectors.detach())
