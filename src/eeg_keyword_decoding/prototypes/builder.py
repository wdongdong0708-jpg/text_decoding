from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.nn import functional as F

from eeg_keyword_decoding.data import (
    LexicalIdentityIndex,
    MasterKeywordIndex,
    SplitViewIndex,
)
from eeg_keyword_decoding.models import ContextTextProjector
from eeg_keyword_decoding.text import (
    ContextSentence,
    ContextWordStore,
    load_context_sentences,
)

from .provenance import file_sha256, module_state_sha256
from .schema import (
    MASTER_KEYWORD_COUNT,
    PROTOTYPE_DIMENSION,
    PrototypeBank,
)


PROTOTYPE_BUILDER_CONFIG_SCHEMA = "train_only_group_balanced_prototype_v1"


@dataclass(frozen=True)
class PrototypeBuilderConfig:
    schema_version: str
    candidate_space: str
    candidate_count: int
    output_dim: int
    source_role: str
    min_train_group_df: int
    occurrence_normalization: str
    aggregation: tuple[str, ...]
    final_normalization: str
    unavailable_vector: str
    refresh_policy: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PrototypeBuilderConfig:
        config = cls(
            schema_version=str(value["schema_version"]),
            candidate_space=str(value["candidate_space"]),
            candidate_count=int(value["candidate_count"]),
            output_dim=int(value["output_dim"]),
            source_role=str(value["source_role"]),
            min_train_group_df=int(value["min_train_group_df"]),
            occurrence_normalization=str(value["occurrence_normalization"]),
            aggregation=tuple(str(item) for item in value["aggregation"]),
            final_normalization=str(value["final_normalization"]),
            unavailable_vector=str(value["unavailable_vector"]),
            refresh_policy=str(value["refresh_policy"]),
        )
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> PrototypeBuilderConfig:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Prototype builder YAML root must be a mapping")
        return cls.from_dict(value)

    def validate(self) -> None:
        if self.schema_version != PROTOTYPE_BUILDER_CONFIG_SCHEMA:
            raise ValueError(
                f"Unsupported prototype schema: {self.schema_version!r}"
            )
        if self.candidate_space != "master_fixed":
            raise ValueError("Prototype candidate space must be fixed Master")
        if self.candidate_count != MASTER_KEYWORD_COUNT:
            raise ValueError("Prototype candidate count must be 247")
        if self.output_dim != PROTOTYPE_DIMENSION:
            raise ValueError("Prototype output dimension must be 256")
        if self.source_role != "train":
            raise ValueError("Prototype source_role must be train")
        if self.min_train_group_df <= 0:
            raise ValueError("min_train_group_df must be positive")
        if self.occurrence_normalization != "l2":
            raise ValueError("Projected occurrences must be L2 normalized")
        if self.aggregation != (
            "within_sentence_keyword_mean",
            "within_sentence_group_mean",
            "equal_sentence_group_mean",
        ):
            raise ValueError("Unexpected prototype aggregation contract")
        if self.final_normalization != "l2":
            raise ValueError("Final prototypes must be L2 normalized")
        if self.unavailable_vector != "zero":
            raise ValueError("Unavailable prototype rows must be zero")
        if self.refresh_policy != "explicit_current_projector":
            raise ValueError("Prototype refresh must be explicit")


def _read_eligibility(
    path: Path,
    *,
    outer_fold: int,
    keyword_index: MasterKeywordIndex,
) -> tuple[torch.Tensor, torch.Tensor]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if int(row["outer_fold"]) == outer_fold
        ]
    if len(rows) != MASTER_KEYWORD_COUNT:
        raise ValueError(
            f"Eligibility fold {outer_fold} must have 247 rows"
        )
    row_by_keyword = {row["keyword_id"]: row for row in rows}
    if set(row_by_keyword) != set(keyword_index.keyword_ids):
        raise ValueError("Eligibility keyword IDs differ from Master")
    sentence_df = torch.tensor(
        [
            int(row_by_keyword[keyword_id]["train_df"])
            for keyword_id in keyword_index.keyword_ids
        ],
        dtype=torch.int64,
    )
    group_df = torch.tensor(
        [
            int(row_by_keyword[keyword_id]["train_group_df"])
            for keyword_id in keyword_index.keyword_ids
        ],
        dtype=torch.int64,
    )
    return sentence_df, group_df


def aggregate_sentence_group_balanced(
    vectors_by_sentence: dict[int, torch.Tensor],
    sentence_group_by_index: dict[int, str],
) -> torch.Tensor:
    """Average sentences inside a normalized group, then groups equally."""

    if not vectors_by_sentence:
        raise ValueError("Cannot aggregate an empty prototype")
    if not set(vectors_by_sentence) <= set(sentence_group_by_index):
        raise ValueError("A prototype sentence is missing its group identity")
    grouped: dict[str, list[torch.Tensor]] = defaultdict(list)
    for text_embedding_idx, vector in vectors_by_sentence.items():
        if vector.ndim != 1:
            raise ValueError("Prototype sentence vectors must be one-dimensional")
        grouped[sentence_group_by_index[text_embedding_idx]].append(vector)
    group_means = [
        torch.stack(vectors).mean(dim=0)
        for _, vectors in sorted(grouped.items())
    ]
    return torch.stack(group_means).mean(dim=0)


def aggregate_within_sentence_keyword(
    projected_occurrences: torch.Tensor,
    word_positions: list[int] | tuple[int, ...],
) -> torch.Tensor:
    """Average repeated occurrences of one keyword inside one sentence."""

    if projected_occurrences.ndim != 2:
        raise ValueError("Projected occurrences must have shape [word,feature]")
    if not word_positions:
        raise ValueError("A sentence keyword needs at least one occurrence")
    positions = torch.tensor(
        word_positions,
        dtype=torch.int64,
        device=projected_occurrences.device,
    )
    if bool((positions < 0).any()) or bool(
        (positions >= projected_occurrences.shape[0]).any()
    ):
        raise ValueError("Sentence keyword position is out of range")
    return projected_occurrences.index_select(0, positions).mean(dim=0)


class TrainOnlyPrototypeBuilder:
    """Rebuild a fold-specific bank from unique inner-train text contexts."""

    def __init__(
        self,
        *,
        config: PrototypeBuilderConfig,
        split_index: SplitViewIndex,
        outer_fold: int,
        keyword_index: MasterKeywordIndex,
        lexical_identity_index: LexicalIdentityIndex,
        sentence_labels_path: str | Path,
        word_occurrences_path: str | Path,
        eligibility_path: str | Path,
        context_store_root: str | Path,
        text_backend: str,
        projection_batch_size: int = 64,
    ) -> None:
        config.validate()
        if outer_fold not in split_index.outer_folds:
            raise ValueError(f"Unknown outer fold: {outer_fold}")
        if text_backend not in {"macbert", "bge_m3"}:
            raise ValueError(f"Unsupported backend: {text_backend!r}")
        if projection_batch_size <= 0:
            raise ValueError("projection_batch_size must be positive")
        if lexical_identity_index.source_fold_sha256 != (
            split_index.fold_source_sha256
        ):
            raise ValueError("Lexical identity fold hash mismatch")

        self.config = config
        self.split_index = split_index
        self.outer_fold = int(outer_fold)
        self.keyword_index = keyword_index
        self.lexical_identity_index = lexical_identity_index
        self.sentence_labels_path = Path(sentence_labels_path)
        self.word_occurrences_path = Path(word_occurrences_path)
        self.eligibility_path = Path(eligibility_path)
        self.context_store_root = Path(context_store_root)
        self.text_backend = text_backend
        self.projection_batch_size = int(projection_batch_size)
        self.sentences = tuple(
            load_context_sentences(
                self.sentence_labels_path,
                self.word_occurrences_path,
            )
        )
        self.sentence_by_idx = {
            sentence.text_embedding_idx: sentence
            for sentence in self.sentences
        }
        self.store = ContextWordStore(
            self.context_store_root,
            verify_hashes=True,
        )
        self.store.audit_against(self.sentences)
        self.source_cache_hash = self.store.metadata.arrays[
            "context_vectors"
        ].sha256
        self.source_cache_metadata_hash = file_sha256(
            self.context_store_root / "metadata.json"
        )
        self.eligibility_hash = file_sha256(self.eligibility_path)

    def _project_batch(
        self,
        projector: ContextTextProjector,
        sentences: list[ContextSentence],
        device: torch.device,
    ) -> list[torch.Tensor]:
        cached = [
            self.store.get_sentence(
                sentence.text_embedding_idx,
                vector_dtype=np.float32,
            ).vectors
            for sentence in sentences
        ]
        maximum_words = max(value.shape[0] for value in cached)
        tail = tuple(cached[0].shape[1:])
        if any(tuple(value.shape[1:]) != tail for value in cached):
            raise ValueError("Context cache contains mixed feature tails")
        context = torch.zeros(
            (len(cached), maximum_words, *tail),
            dtype=torch.float32,
            device=device,
        )
        mask = torch.zeros(
            (len(cached), maximum_words),
            dtype=torch.bool,
            device=device,
        )
        for index, value in enumerate(cached):
            length = value.shape[0]
            context[index, :length] = torch.from_numpy(
                np.array(value, dtype=np.float32, copy=True)
            ).to(device)
            mask[index, :length] = True
        output = projector(
            context_words=context,
            word_mask=mask,
            backend=self.text_backend,
        ).sequence
        return [
            output[index, : sentence.word_count].detach().cpu().float()
            for index, sentence in enumerate(sentences)
        ]

    def build(
        self,
        projector: ContextTextProjector,
        *,
        role: str = "train",
    ) -> PrototypeBank:
        if role != "train":
            raise ValueError(
                "Prototype builder accepts only role='train'; validation/test "
                "contributors are forbidden"
            )
        if projector.config.backend != self.text_backend:
            raise ValueError("Projector backend does not match prototype backend")
        projector.config.validate_cache_metadata(
            self.store.metadata.to_dict()
        )
        if projector.config.expected_context_vectors_sha256 != (
            self.source_cache_hash
        ):
            raise ValueError("Projector/cache vector hash mismatch")

        train_indices = self.split_index.sentence_indices_for(
            self.outer_fold,
            "train",
        )
        validation_indices = set(
            self.split_index.sentence_indices_for(
                self.outer_fold,
                "validation",
            )
        )
        test_indices = set(
            self.split_index.sentence_indices_for(
                self.outer_fold,
                "test",
            )
        )
        contributors = set(train_indices)
        if contributors & validation_indices:
            raise RuntimeError("Validation contexts leaked into prototype source")
        if contributors & test_indices:
            raise RuntimeError("Test contexts leaked into prototype source")
        if len(contributors) != len(train_indices):
            raise RuntimeError("Duplicate train text_embedding_idx contributor")

        device = next(projector.parameters()).device
        was_training = projector.training
        projected_by_sentence: dict[int, torch.Tensor] = {}
        projector.eval()
        try:
            with torch.inference_mode():
                for start in range(0, len(train_indices), self.projection_batch_size):
                    indices = train_indices[
                        start : start + self.projection_batch_size
                    ]
                    sentences = [self.sentence_by_idx[index] for index in indices]
                    values = self._project_batch(projector, sentences, device)
                    projected_by_sentence.update(
                        {
                            index: value
                            for index, value in zip(
                                indices,
                                values,
                                strict=True,
                            )
                        }
                    )
        finally:
            projector.train(was_training)
        if set(projected_by_sentence) != contributors:
            raise RuntimeError("Some train contexts were not projected")

        sentence_vectors: list[dict[int, torch.Tensor]] = [
            {} for _ in range(MASTER_KEYWORD_COUNT)
        ]
        sentence_groups: dict[int, str] = {}
        for text_embedding_idx in train_indices:
            assignment = self.split_index.assignment(
                self.outer_fold,
                text_embedding_idx,
            )
            if assignment.role != "train":
                raise RuntimeError("Non-train assignment reached builder")
            sentence_groups[text_embedding_idx] = assignment.sentence_group_id
            sentence = self.sentence_by_idx[text_embedding_idx]
            projected = F.normalize(
                projected_by_sentence[text_embedding_idx],
                p=2,
                dim=-1,
                eps=1e-8,
            )
            positions_by_keyword: dict[int, list[int]] = defaultdict(list)
            for occurrence in sentence.occurrences:
                keyword_index = self.keyword_index.index_or_minus_one(
                    occurrence.keyword_id
                )
                if keyword_index >= 0:
                    positions_by_keyword[keyword_index].append(
                        occurrence.word_position
                    )
            for keyword_index, positions in positions_by_keyword.items():
                sentence_vectors[keyword_index][text_embedding_idx] = (
                    aggregate_within_sentence_keyword(
                        projected,
                        positions,
                    )
                )

        computed_sentence_df = torch.tensor(
            [len(values) for values in sentence_vectors],
            dtype=torch.int64,
        )
        group_vectors: list[dict[str, list[torch.Tensor]]] = [
            defaultdict(list) for _ in range(MASTER_KEYWORD_COUNT)
        ]
        for keyword_index, values in enumerate(sentence_vectors):
            for text_embedding_idx, vector in values.items():
                group_vectors[keyword_index][
                    sentence_groups[text_embedding_idx]
                ].append(vector)
        computed_group_df = torch.tensor(
            [len(values) for values in group_vectors],
            dtype=torch.int64,
        )
        expected_sentence_df, expected_group_df = _read_eligibility(
            self.eligibility_path,
            outer_fold=self.outer_fold,
            keyword_index=self.keyword_index,
        )
        if not torch.equal(computed_sentence_df, expected_sentence_df):
            mismatch = torch.nonzero(
                computed_sentence_df != expected_sentence_df
            ).flatten()
            raise ValueError(
                "Prototype train sentence DF disagrees with frozen "
                f"eligibility at rows {mismatch[:10].tolist()}"
            )
        if not torch.equal(computed_group_df, expected_group_df):
            mismatch = torch.nonzero(
                computed_group_df != expected_group_df
            ).flatten()
            raise ValueError(
                "Prototype train group DF disagrees with frozen eligibility "
                f"at rows {mismatch[:10].tolist()}"
            )

        available = computed_group_df >= self.config.min_train_group_df
        vectors = torch.zeros(
            (MASTER_KEYWORD_COUNT, PROTOTYPE_DIMENSION),
            dtype=torch.float32,
        )
        for keyword_index in torch.nonzero(available).flatten().tolist():
            vector = aggregate_sentence_group_balanced(
                sentence_vectors[keyword_index],
                sentence_groups,
            )
            vectors[keyword_index] = F.normalize(
                vector,
                p=2,
                dim=0,
                eps=1e-8,
            )

        projector_hash = module_state_sha256(projector)
        metadata = {
            "schema_version": self.config.schema_version,
            "source_role": "train",
            "outer_fold": self.outer_fold,
            "text_backend": self.text_backend,
            "min_train_group_df": self.config.min_train_group_df,
            "aggregation": list(self.config.aggregation),
            "occurrence_normalization": self.config.occurrence_normalization,
            "final_normalization": self.config.final_normalization,
            "projector_refresh": {
                "policy": self.config.refresh_policy,
                "bank_is_detached": True,
                "prototype_loss_bank_gradient": False,
                "current_alignment_path_updates_text_projector": True,
            },
            "contributors": {
                "text_embedding_indices": list(train_indices),
                "unique_text_embedding_indices": len(contributors),
                "validation_intersection": [],
                "test_intersection": [],
                "eeg_views_used": False,
                "each_text_embedding_idx_counted_once": True,
                "sentence_group_count": len(set(sentence_groups.values())),
            },
            "hashes": {
                "word_occurrences": file_sha256(
                    self.word_occurrences_path
                ),
                "folds": self.split_index.fold_source_sha256,
                "eligibility": self.eligibility_hash,
                "context_vectors": self.source_cache_hash,
                "context_cache_metadata": (
                    self.source_cache_metadata_hash
                ),
                "lexical_mapping": (
                    self.lexical_identity_index.mapping_sha256
                ),
                "projector_state": projector_hash,
            },
        }
        bank = PrototypeBank(
            vectors=vectors.detach(),
            available_mask=available,
            keyword_ids=self.keyword_index.keyword_ids,
            train_sentence_df=computed_sentence_df,
            train_group_df=computed_group_df,
            outer_fold=self.outer_fold,
            text_backend=self.text_backend,
            projector_state_hash=projector_hash,
            source_cache_hash=self.source_cache_hash,
            source_cache_metadata_hash=self.source_cache_metadata_hash,
            fold_hash=self.split_index.fold_source_sha256,
            eligibility_hash=self.eligibility_hash,
            lexical_mapping_hash=self.lexical_identity_index.mapping_sha256,
            metadata=metadata,
        )
        bank.validate()
        return bank
