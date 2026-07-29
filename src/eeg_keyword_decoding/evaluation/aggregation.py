from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from eeg_keyword_decoding.prototypes.schema import MASTER_KEYWORD_COUNT


@dataclass(frozen=True)
class KeywordPredictionTable:
    level: str
    scores: torch.Tensor
    labels: torch.Tensor
    unit_ids: tuple[str, ...]
    text_embedding_indices: tuple[int, ...]
    sentence_group_ids: tuple[str, ...]
    member_counts: torch.Tensor
    source_view_ids: tuple[tuple[str, ...], ...] = ()
    source_subjects: tuple[tuple[str, ...], ...] = ()

    def validate(self) -> None:
        rows = self.scores.shape[0]
        if self.scores.ndim != 2 or self.scores.shape[1] != MASTER_KEYWORD_COUNT:
            raise ValueError("scores must have fixed shape [N,247]")
        if self.labels.shape != self.scores.shape or self.labels.dtype != torch.bool:
            raise ValueError("labels must be bool with the same shape as scores")
        if not bool(torch.isfinite(self.scores).all()):
            raise ValueError("scores contain NaN or Inf")
        if any(
            len(values) != rows
            for values in (
                self.unit_ids,
                self.text_embedding_indices,
                self.sentence_group_ids,
            )
        ):
            raise ValueError("Prediction metadata length mismatch")
        if self.member_counts.shape != (rows,) or self.member_counts.dtype != torch.int64:
            raise ValueError("member_counts must be int64 with shape [N]")
        if bool((self.member_counts <= 0).any()):
            raise ValueError("member_counts must be positive")
        for name, values in (
            ("source_view_ids", self.source_view_ids),
            ("source_subjects", self.source_subjects),
        ):
            if values and (
                len(values) != rows
                or any(len(value) == 0 for value in values)
            ):
                raise ValueError(f"{name} must contain a non-empty tuple per row")


def _presence_matrix(
    rows: int,
    present_keyword_indices: Sequence[Iterable[int]],
    *,
    device: torch.device,
) -> torch.Tensor:
    if len(present_keyword_indices) != rows:
        raise ValueError("Keyword-label row count mismatch")
    labels = torch.zeros((rows, MASTER_KEYWORD_COUNT), dtype=torch.bool, device=device)
    for row_index, indices in enumerate(present_keyword_indices):
        for keyword_index in indices:
            value = int(keyword_index)
            if value < 0 or value >= MASTER_KEYWORD_COUNT:
                raise ValueError(f"Keyword index outside Master vocabulary: {value}")
            labels[row_index, value] = True
    return labels


def build_view_prediction_table(
    *,
    scores: torch.Tensor,
    present_keyword_indices: Sequence[Iterable[int]],
    view_ids: Sequence[str],
    text_embedding_indices: Sequence[int] | torch.Tensor,
    sentence_group_ids: Sequence[str],
    subjects: Sequence[str] | None = None,
) -> KeywordPredictionTable:
    if scores.ndim != 2 or scores.shape[1] != MASTER_KEYWORD_COUNT:
        raise ValueError("scores must have fixed shape [B,247]")
    rows = scores.shape[0]
    text_indices = tuple(int(value) for value in text_embedding_indices)
    table = KeywordPredictionTable(
        level="view",
        scores=scores,
        labels=_presence_matrix(
            rows, present_keyword_indices, device=scores.device
        ),
        unit_ids=tuple(str(value) for value in view_ids),
        text_embedding_indices=text_indices,
        sentence_group_ids=tuple(str(value) for value in sentence_group_ids),
        member_counts=torch.ones(rows, dtype=torch.int64, device=scores.device),
        source_view_ids=tuple((str(value),) for value in view_ids),
        source_subjects=(
            tuple((str(value),) for value in subjects)
            if subjects is not None
            else ()
        ),
    )
    table.validate()
    if len(set(table.unit_ids)) != rows:
        raise ValueError("EEG view IDs must be unique")
    return table


def _ordered_groups(values: Sequence[object]) -> tuple[object, ...]:
    return tuple(dict.fromkeys(values))


def aggregate_by_sentence_occurrence(
    view_table: KeywordPredictionTable,
) -> KeywordPredictionTable:
    view_table.validate()
    if view_table.level != "view":
        raise ValueError("Expected a view-level table")
    ordered = _ordered_groups(view_table.text_embedding_indices)
    scores: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    group_ids: list[str] = []
    counts: list[int] = []
    source_views: list[tuple[str, ...]] = []
    source_subjects: list[tuple[str, ...]] = []
    for text_index in ordered:
        members = [
            index
            for index, value in enumerate(view_table.text_embedding_indices)
            if value == text_index
        ]
        member = torch.tensor(members, device=view_table.scores.device)
        member_labels = view_table.labels.index_select(0, member)
        if not bool((member_labels == member_labels[0]).all()):
            raise ValueError(
                f"Keyword labels conflict for text_embedding_idx={text_index}"
            )
        member_groups = {view_table.sentence_group_ids[index] for index in members}
        if len(member_groups) != 1:
            raise ValueError(
                f"sentence_group_id conflicts for text_embedding_idx={text_index}"
            )
        scores.append(view_table.scores.index_select(0, member).mean(dim=0))
        labels.append(member_labels[0])
        group_ids.append(next(iter(member_groups)))
        counts.append(len(members))
        if view_table.source_view_ids:
            source_views.append(
                tuple(
                    value
                    for index in members
                    for value in view_table.source_view_ids[index]
                )
            )
        if view_table.source_subjects:
            source_subjects.append(
                tuple(
                    value
                    for index in members
                    for value in view_table.source_subjects[index]
                )
            )
    table = KeywordPredictionTable(
        level="sentence_occurrence",
        scores=torch.stack(scores),
        labels=torch.stack(labels),
        unit_ids=tuple(f"text_embedding_idx:{value}" for value in ordered),
        text_embedding_indices=tuple(int(value) for value in ordered),
        sentence_group_ids=tuple(group_ids),
        member_counts=torch.tensor(
            counts, dtype=torch.int64, device=view_table.scores.device
        ),
        source_view_ids=tuple(source_views),
        source_subjects=tuple(source_subjects),
    )
    table.validate()
    return table


def aggregate_by_context_group(
    occurrence_table: KeywordPredictionTable,
) -> KeywordPredictionTable:
    occurrence_table.validate()
    if occurrence_table.level != "sentence_occurrence":
        raise ValueError("Expected a sentence-occurrence table")
    ordered = _ordered_groups(occurrence_table.sentence_group_ids)
    scores: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    text_indices: list[int] = []
    counts: list[int] = []
    source_views: list[tuple[str, ...]] = []
    source_subjects: list[tuple[str, ...]] = []
    for group_id in ordered:
        members = [
            index
            for index, value in enumerate(occurrence_table.sentence_group_ids)
            if value == group_id
        ]
        member = torch.tensor(members, device=occurrence_table.scores.device)
        member_labels = occurrence_table.labels.index_select(0, member)
        if not bool((member_labels == member_labels[0]).all()):
            raise ValueError(f"Keyword labels conflict for sentence_group_id={group_id}")
        # Average occurrences, not their underlying view counts.
        scores.append(occurrence_table.scores.index_select(0, member).mean(dim=0))
        labels.append(member_labels[0])
        text_indices.append(occurrence_table.text_embedding_indices[members[0]])
        counts.append(len(members))
        if occurrence_table.source_view_ids:
            source_views.append(
                tuple(
                    value
                    for index in members
                    for value in occurrence_table.source_view_ids[index]
                )
            )
        if occurrence_table.source_subjects:
            source_subjects.append(
                tuple(
                    value
                    for index in members
                    for value in occurrence_table.source_subjects[index]
                )
            )
    table = KeywordPredictionTable(
        level="context_group",
        scores=torch.stack(scores),
        labels=torch.stack(labels),
        unit_ids=tuple(f"sentence_group_id:{value}" for value in ordered),
        text_embedding_indices=tuple(text_indices),
        sentence_group_ids=tuple(str(value) for value in ordered),
        member_counts=torch.tensor(
            counts, dtype=torch.int64, device=occurrence_table.scores.device
        ),
        source_view_ids=tuple(source_views),
        source_subjects=tuple(source_subjects),
    )
    table.validate()
    return table
