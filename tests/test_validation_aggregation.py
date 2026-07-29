from __future__ import annotations

import pytest
import torch

from eeg_keyword_decoding.evaluation import (
    aggregate_by_context_group,
    aggregate_by_sentence_occurrence,
    build_view_prediction_table,
)


def _scores(values: list[float]) -> torch.Tensor:
    result = torch.zeros(len(values), 247)
    result[:, 0] = torch.tensor(values)
    return result


def test_view_to_occurrence_to_context_group_matches_manual_example() -> None:
    view = build_view_prediction_table(
        scores=_scores([1.0, 3.0, 5.0, 9.0, 11.0]),
        present_keyword_indices=((0,), (0,), (0,), (0,), (0,)),
        view_ids=("v0", "v1", "v2", "v3", "v4"),
        text_embedding_indices=(10, 10, 11, 12, 12),
        sentence_group_ids=("g", "g", "g", "h", "h"),
        subjects=("s1", "s2", "s3", "s4", "s5"),
    )
    occurrence = aggregate_by_sentence_occurrence(view)
    assert occurrence.text_embedding_indices == (10, 11, 12)
    assert occurrence.member_counts.tolist() == [2, 1, 2]
    assert occurrence.scores[:, 0].tolist() == [2.0, 5.0, 10.0]
    assert occurrence.source_view_ids[0] == ("v0", "v1")
    assert occurrence.source_subjects[0] == ("s1", "s2")

    group = aggregate_by_context_group(occurrence)
    assert group.sentence_group_ids == ("g", "h")
    assert group.member_counts.tolist() == [2, 1]
    assert group.scores[:, 0].tolist() == [3.5, 10.0]
    assert group.source_view_ids[0] == ("v0", "v1", "v2")


def test_view_count_does_not_reweight_occurrences() -> None:
    view = build_view_prediction_table(
        scores=_scores([2.0, 2.0, 2.0, 8.0]),
        present_keyword_indices=((0,),) * 4,
        view_ids=("a", "b", "c", "d"),
        text_embedding_indices=(1, 1, 1, 2),
        sentence_group_ids=("same",) * 4,
    )
    group = aggregate_by_context_group(aggregate_by_sentence_occurrence(view))
    assert group.scores[0, 0].item() == 5.0


def test_label_conflicts_raise_at_both_aggregation_boundaries() -> None:
    view = build_view_prediction_table(
        scores=_scores([1.0, 2.0]),
        present_keyword_indices=((0,), (1,)),
        view_ids=("a", "b"),
        text_embedding_indices=(1, 1),
        sentence_group_ids=("g", "g"),
    )
    with pytest.raises(ValueError, match="labels conflict"):
        aggregate_by_sentence_occurrence(view)

    occurrence = aggregate_by_sentence_occurrence(
        build_view_prediction_table(
            scores=_scores([1.0, 2.0]),
            present_keyword_indices=((0,), (1,)),
            view_ids=("a", "b"),
            text_embedding_indices=(1, 2),
            sentence_group_ids=("g", "g"),
        )
    )
    with pytest.raises(ValueError, match="labels conflict"):
        aggregate_by_context_group(occurrence)
