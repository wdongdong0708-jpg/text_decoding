"""Text-free, fixed-lexicon keyword evaluation."""

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
    load_fold_keyword_eligibility,
)
from .prototype_scorer import PrototypeScoreOutput, TextFreePrototypeScorer
from .validation import TextFreeValidationResult, run_text_free_validation

__all__ = [
    "FoldKeywordEligibility",
    "KeywordMetricReport",
    "KeywordPredictionTable",
    "PrototypeScoreOutput",
    "TextFreePrototypeScorer",
    "TextFreeValidationResult",
    "aggregate_by_context_group",
    "aggregate_by_sentence_occurrence",
    "build_view_prediction_table",
    "compute_keyword_auprc",
    "load_fold_keyword_eligibility",
    "run_text_free_validation",
]
