from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from eeg_keyword_decoding.data.keyword_index import MasterKeywordIndex
from eeg_keyword_decoding.prototypes.schema import MASTER_KEYWORD_COUNT

from .aggregation import KeywordPredictionTable

KeywordTier = Literal["core", "main", "extended", "story_local"]


@dataclass(frozen=True)
class FoldKeywordEligibility:
    outer_fold: int
    keyword_ids: tuple[str, ...]
    core_mask: torch.Tensor
    main_mask: torch.Tensor
    extended_mask: torch.Tensor
    story_local_mask: torch.Tensor
    group_eligible_mask: torch.Tensor
    validation_group_df: torch.Tensor

    def tier_mask(self, tier: KeywordTier) -> torch.Tensor:
        return {
            "core": self.core_mask,
            "main": self.main_mask,
            "extended": self.extended_mask,
            "story_local": self.story_local_mask,
        }[tier]

    def validate(self) -> None:
        if self.outer_fold not in range(5):
            raise ValueError("outer_fold must be in [0,4]")
        if len(self.keyword_ids) != MASTER_KEYWORD_COUNT:
            raise ValueError("Eligibility must retain all 247 Master rows")
        for name, value in (
            ("core_mask", self.core_mask),
            ("main_mask", self.main_mask),
            ("extended_mask", self.extended_mask),
            ("story_local_mask", self.story_local_mask),
            ("group_eligible_mask", self.group_eligible_mask),
        ):
            if value.shape != (MASTER_KEYWORD_COUNT,) or value.dtype != torch.bool:
                raise ValueError(f"{name} must be bool with shape [247]")
        if (
            self.validation_group_df.shape != (MASTER_KEYWORD_COUNT,)
            or self.validation_group_df.dtype != torch.int64
        ):
            raise ValueError("validation_group_df must be int64 with shape [247]")


@dataclass(frozen=True)
class KeywordMetricReport:
    tier: KeywordTier
    level: str
    macro_auprc: float
    valid_keyword_count: int
    total_tier_keyword_count: int
    eligible_keyword_count: int
    average_precision: torch.Tensor
    positive_counts: torch.Tensor
    negative_counts: torch.Tensor
    validity_reasons: tuple[str, ...]

    def to_summary(self) -> dict[str, object]:
        return {
            "tier": self.tier,
            "level": self.level,
            "macro_auprc": self.macro_auprc,
            "valid_keyword_count": self.valid_keyword_count,
            "total_tier_keyword_count": self.total_tier_keyword_count,
            "eligible_keyword_count": self.eligible_keyword_count,
            "positive_counts": self.positive_counts.tolist(),
            "negative_counts": self.negative_counts.tolist(),
            "validity_reasons": list(self.validity_reasons),
        }


def load_fold_keyword_eligibility(
    path: str | Path,
    *,
    outer_fold: int,
    keyword_index: MasterKeywordIndex,
) -> FoldKeywordEligibility:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    rows = [row for row in all_rows if int(row["outer_fold"]) == outer_fold]
    by_id = {row["keyword_id"]: row for row in rows}
    if len(rows) != MASTER_KEYWORD_COUNT or set(by_id) != set(keyword_index.keyword_ids):
        raise ValueError("Fold eligibility must contain the exact 247 Master keywords")

    ordered = [by_id[keyword_id] for keyword_id in keyword_index.keyword_ids]

    def flags(name: str) -> torch.Tensor:
        return torch.tensor(
            [row[name].strip().lower() == "true" for row in ordered],
            dtype=torch.bool,
        )

    result = FoldKeywordEligibility(
        outer_fold=outer_fold,
        keyword_ids=keyword_index.keyword_ids,
        core_mask=flags("include_core"),
        main_mask=flags("include_main"),
        extended_mask=flags("include_extended"),
        story_local_mask=flags("story_local_flag"),
        group_eligible_mask=flags("eligible_group_20_5_5"),
        validation_group_df=torch.tensor(
            [int(row["validation_group_df"]) for row in ordered],
            dtype=torch.int64,
        ),
    )
    result.validate()
    return result


def compute_keyword_auprc(
    table: KeywordPredictionTable,
    *,
    eligibility: FoldKeywordEligibility,
    available_mask: torch.Tensor,
    tier: KeywordTier = "core",
) -> KeywordMetricReport:
    table.validate()
    eligibility.validate()
    if available_mask.shape != (MASTER_KEYWORD_COUNT,) or available_mask.dtype != torch.bool:
        raise ValueError("available_mask must be bool with shape [247]")
    tier_mask = eligibility.tier_mask(tier).cpu()
    eligible = tier_mask & eligibility.group_eligible_mask.cpu()
    available = available_mask.detach().cpu()
    labels = table.labels.detach().cpu().numpy().astype(bool, copy=False)
    scores = table.scores.detach().float().cpu().numpy()

    ap = np.full(MASTER_KEYWORD_COUNT, np.nan, dtype=np.float64)
    positive = labels.sum(axis=0, dtype=np.int64)
    negative = labels.shape[0] - positive
    reasons: list[str] = []
    for column in range(MASTER_KEYWORD_COUNT):
        if not bool(tier_mask[column]):
            reasons.append("outside_tier")
        elif not bool(eligible[column]):
            reasons.append("ineligible_group_20_5_5")
        elif not bool(available[column]):
            reasons.append("prototype_unavailable")
        elif positive[column] == 0:
            reasons.append("no_positive")
        elif negative[column] == 0:
            reasons.append("no_negative")
        else:
            ap[column] = float(average_precision_score(labels[:, column], scores[:, column]))
            reasons.append("valid")

    valid = np.isfinite(ap) & eligible.numpy() & available.numpy()
    macro = float(np.mean(ap[valid])) if bool(valid.any()) else float("nan")
    return KeywordMetricReport(
        tier=tier,
        level=table.level,
        macro_auprc=macro,
        valid_keyword_count=int(valid.sum()),
        total_tier_keyword_count=int(tier_mask.sum().item()),
        eligible_keyword_count=int(eligible.sum().item()),
        average_precision=torch.from_numpy(ap),
        positive_counts=torch.from_numpy(positive.copy()),
        negative_counts=torch.from_numpy(negative.copy()),
        validity_reasons=tuple(reasons),
    )
