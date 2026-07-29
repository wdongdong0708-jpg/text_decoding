from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from eeg_keyword_decoding.data import ContextEEGBatch
from eeg_keyword_decoding.prototypes import PrototypeBank

from .aggregation import (
    KeywordPredictionTable,
    aggregate_by_context_group,
    aggregate_by_sentence_occurrence,
    build_view_prediction_table,
)
from .keyword_metrics import (
    FoldKeywordEligibility,
    KeywordMetricReport,
    compute_keyword_auprc,
)
from .prototype_scorer import TextFreePrototypeScorer


@dataclass(frozen=True)
class TextFreeValidationResult:
    tables: dict[str, KeywordPredictionTable]
    reports: dict[str, KeywordMetricReport]
    metrics: dict[str, float]
    scorer_temperature: float
    prototype_available_count: int


def run_text_free_validation(
    *,
    eeg_encoder: nn.Module,
    validation_loader: Iterable[ContextEEGBatch],
    prototype_bank: PrototypeBank,
    scorer: TextFreePrototypeScorer,
    eligibility: FoldKeywordEligibility,
    device: torch.device | str,
    max_batches: int | None = None,
) -> TextFreeValidationResult:
    target = torch.device(device)
    encoder_was_training = eeg_encoder.training
    eeg_encoder.eval()
    bank = prototype_bank.to(target)
    score_parts: list[torch.Tensor] = []
    present: list[torch.Tensor] = []
    view_ids: list[str] = []
    text_indices: list[int] = []
    group_ids: list[str] = []
    subjects: list[str] = []
    try:
        with torch.inference_mode():
            for batch_index, batch in enumerate(validation_loader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                if batch.context_words is not None:
                    raise RuntimeError(
                        "Validation batch contains forbidden context vectors"
                    )
                if set(batch.roles) != {"validation"}:
                    raise RuntimeError("Validation loader yielded a non-validation role")
                moved = batch.to(target, non_blocking=True)
                encoded = eeg_encoder(
                    eeg=moved.eeg,
                    eeg_mask=moved.eeg_mask,
                    subject_indices=moved.subject_indices,
                )
                output = scorer(encoded.sequence, encoded.mask, bank)
                score_parts.append(output.scores.cpu())
                present.extend(value.cpu() for value in batch.present_keyword_indices)
                view_ids.extend(batch.eeg_view_ids)
                text_indices.extend(
                    int(value) for value in batch.text_embedding_indices.tolist()
                )
                group_ids.extend(batch.sentence_group_ids)
                subjects.extend(batch.subjects)
    finally:
        eeg_encoder.train(encoder_was_training)
    if not score_parts:
        raise ValueError("Validation loader produced no batches")

    view = build_view_prediction_table(
        scores=torch.cat(score_parts, dim=0),
        present_keyword_indices=present,
        view_ids=view_ids,
        text_embedding_indices=text_indices,
        sentence_group_ids=group_ids,
        subjects=subjects,
    )
    occurrence = aggregate_by_sentence_occurrence(view)
    context_group = aggregate_by_context_group(occurrence)
    tables = {
        "view": view,
        "sentence_occurrence": occurrence,
        "context_group": context_group,
    }
    reports: dict[str, KeywordMetricReport] = {}
    metrics: dict[str, float] = {}
    for level, table in tables.items():
        for tier in ("core", "main", "extended", "story_local"):
            report = compute_keyword_auprc(
                table,
                eligibility=eligibility,
                available_mask=prototype_bank.available_mask,
                tier=tier,  # type: ignore[arg-type]
            )
            key = f"validation/{tier}/{level}/macro_auprc"
            reports[f"{tier}/{level}"] = report
            metrics[key] = report.macro_auprc
    return TextFreeValidationResult(
        tables=tables,
        reports=reports,
        metrics=metrics,
        scorer_temperature=scorer.temperature,
        prototype_available_count=prototype_bank.available_count,
    )
