from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
from torch.nn import functional as F


PROTOTYPE_BANK_SCHEMA_VERSION = "train_only_group_balanced_prototype_v1"
MASTER_KEYWORD_COUNT = 247
PROTOTYPE_DIMENSION = 256


@dataclass(frozen=True)
class PrototypeBank:
    vectors: torch.Tensor
    available_mask: torch.Tensor
    keyword_ids: tuple[str, ...]
    train_sentence_df: torch.Tensor
    train_group_df: torch.Tensor
    outer_fold: int
    text_backend: str
    projector_state_hash: str
    source_cache_hash: str
    source_cache_metadata_hash: str
    fold_hash: str
    eligibility_hash: str
    lexical_mapping_hash: str
    metadata: dict[str, Any]

    def validate(self) -> None:
        if self.vectors.shape != (
            MASTER_KEYWORD_COUNT,
            PROTOTYPE_DIMENSION,
        ):
            raise ValueError(
                "Prototype vectors must have fixed shape [247,256], got "
                f"{tuple(self.vectors.shape)}"
            )
        if self.vectors.dtype != torch.float32:
            raise ValueError("Prototype vectors must be float32")
        if self.available_mask.shape != (MASTER_KEYWORD_COUNT,):
            raise ValueError("available_mask must have shape [247]")
        if self.available_mask.dtype != torch.bool:
            raise ValueError("available_mask must be torch.bool")
        for name, value in (
            ("train_sentence_df", self.train_sentence_df),
            ("train_group_df", self.train_group_df),
        ):
            if value.shape != (MASTER_KEYWORD_COUNT,):
                raise ValueError(f"{name} must have shape [247]")
            if value.dtype != torch.int64:
                raise ValueError(f"{name} must be int64")
            if bool((value < 0).any()):
                raise ValueError(f"{name} must be non-negative")
        if len(self.keyword_ids) != MASTER_KEYWORD_COUNT:
            raise ValueError("Prototype bank must retain all 247 keyword IDs")
        if len(set(self.keyword_ids)) != MASTER_KEYWORD_COUNT:
            raise ValueError("Prototype keyword IDs must be unique")
        if self.outer_fold not in range(5):
            raise ValueError("outer_fold must be in [0,4]")
        if self.text_backend not in {"macbert", "bge_m3"}:
            raise ValueError(f"Unsupported text backend: {self.text_backend!r}")
        for name, value in (
            ("projector_state_hash", self.projector_state_hash),
            ("source_cache_hash", self.source_cache_hash),
            ("source_cache_metadata_hash", self.source_cache_metadata_hash),
            ("fold_hash", self.fold_hash),
            ("eligibility_hash", self.eligibility_hash),
            ("lexical_mapping_hash", self.lexical_mapping_hash),
        ):
            if len(value) != 64:
                raise ValueError(f"{name} must be a SHA256 digest")
        if not bool(torch.isfinite(self.vectors).all()):
            raise ValueError("Prototype vectors contain NaN or Inf")
        available_norms = self.vectors[self.available_mask].norm(dim=-1)
        if available_norms.numel() == 0:
            raise ValueError("Prototype bank has no available rows")
        if not torch.allclose(
            available_norms,
            torch.ones_like(available_norms),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise ValueError("Available prototype rows must have unit norm")
        if bool((self.vectors[~self.available_mask] != 0).any()):
            raise ValueError("Unavailable prototype rows must be zero")

    @property
    def available_count(self) -> int:
        return int(self.available_mask.sum().item())

    def detached(self) -> PrototypeBank:
        return replace(
            self,
            vectors=self.vectors.detach(),
            available_mask=self.available_mask.detach(),
            train_sentence_df=self.train_sentence_df.detach(),
            train_group_df=self.train_group_df.detach(),
        )

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> PrototypeBank:
        target = torch.device(device)
        bank = replace(
            self,
            vectors=self.vectors.to(
                target,
                dtype=torch.float32,
                non_blocking=non_blocking,
            ),
            available_mask=self.available_mask.to(
                target,
                non_blocking=non_blocking,
            ),
            train_sentence_df=self.train_sentence_df.to(
                target,
                non_blocking=non_blocking,
            ),
            train_group_df=self.train_group_df.to(
                target,
                non_blocking=non_blocking,
            ),
        )
        bank.validate()
        return bank

    def normalized_vectors(self) -> torch.Tensor:
        return F.normalize(self.vectors, p=2, dim=-1, eps=1e-8)
