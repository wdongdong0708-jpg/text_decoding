from __future__ import annotations

import math
from pathlib import Path

import torch
from sklearn.metrics import average_precision_score

from eeg_keyword_decoding.data import build_master_keyword_index
from eeg_keyword_decoding.evaluation import (
    FoldKeywordEligibility,
    KeywordPredictionTable,
    compute_keyword_auprc,
    load_fold_keyword_eligibility,
)


def _eligibility() -> FoldKeywordEligibility:
    core = torch.zeros(247, dtype=torch.bool)
    core[:3] = True
    eligible = torch.zeros(247, dtype=torch.bool)
    eligible[:3] = True
    return FoldKeywordEligibility(
        outer_fold=0,
        keyword_ids=tuple(f"kw-{i}" for i in range(247)),
        core_mask=core,
        main_mask=core.clone(),
        extended_mask=core.clone(),
        story_local_mask=torch.zeros(247, dtype=torch.bool),
        group_eligible_mask=eligible,
        validation_group_df=torch.ones(247, dtype=torch.int64),
    )


def test_per_keyword_ap_macro_and_nan_reasons() -> None:
    scores = torch.zeros(4, 247)
    scores[:, 0] = torch.tensor([0.9, 0.1, 0.8, 0.2])
    scores[:, 1] = torch.tensor([0.4, 0.3, 0.2, 0.1])
    labels = torch.zeros(4, 247, dtype=torch.bool)
    labels[:, 0] = torch.tensor([True, False, True, False])
    labels[:, 2] = True
    table = KeywordPredictionTable(
        level="context_group",
        scores=scores,
        labels=labels,
        unit_ids=("a", "b", "c", "d"),
        text_embedding_indices=(1, 2, 3, 4),
        sentence_group_ids=("a", "b", "c", "d"),
        member_counts=torch.ones(4, dtype=torch.int64),
    )
    report = compute_keyword_auprc(
        table,
        eligibility=_eligibility(),
        available_mask=torch.ones(247, dtype=torch.bool),
    )
    expected = average_precision_score(labels[:, 0].numpy(), scores[:, 0].numpy())
    assert report.average_precision[0].item() == expected
    assert math.isnan(report.average_precision[1].item())
    assert report.validity_reasons[1] == "no_positive"
    assert math.isnan(report.average_precision[2].item())
    assert report.validity_reasons[2] == "no_negative"
    assert report.valid_keyword_count == 1
    assert report.macro_auprc == expected


def test_unavailable_or_ineligible_columns_do_not_shrink_score_space() -> None:
    scores = torch.zeros(2, 247)
    labels = torch.zeros(2, 247, dtype=torch.bool)
    labels[0, 0] = True
    table = KeywordPredictionTable(
        level="view",
        scores=scores,
        labels=labels,
        unit_ids=("a", "b"),
        text_embedding_indices=(1, 2),
        sentence_group_ids=("a", "b"),
        member_counts=torch.ones(2, dtype=torch.int64),
    )
    available = torch.ones(247, dtype=torch.bool)
    available[0] = False
    report = compute_keyword_auprc(
        table, eligibility=_eligibility(), available_mask=available
    )
    assert table.scores.shape[1] == 247
    assert report.validity_reasons[0] == "prototype_unavailable"


def test_real_fold_eligibility_retains_master_and_core_sizes() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = root / "data" / "protocols" / "littleprince_hf_v1"
    keyword_index = build_master_keyword_index(
        protocol / "littleprince_hf_lexicon_v1.csv"
    )
    eligibility = load_fold_keyword_eligibility(
        protocol / "littleprince_keyword_fold_eligibility_v1.csv",
        outer_fold=0,
        keyword_index=keyword_index,
    )
    assert len(eligibility.keyword_ids) == 247
    assert eligibility.core_mask.sum().item() == 33
    assert (eligibility.core_mask & eligibility.group_eligible_mask).sum().item() == 33
