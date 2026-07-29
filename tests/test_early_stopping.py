import pytest

from eeg_keyword_decoding.training import EarlyStopping


def test_early_stopping_uses_only_preregistered_metric_and_keeps_earlier_tie() -> None:
    stopping = EarlyStopping(patience=1, min_delta=0.001)
    metric = "validation/core/context_group/macro_auprc"
    assert stopping.update({metric: 0.2, "outer_test/auprc": 1.0}, epoch=0)
    assert not stopping.update({metric: 0.2, "outer_test/auprc": 0.0}, epoch=1)
    assert stopping.best_epoch == 0
    assert stopping.should_stop


def test_early_stopping_rejects_other_primary_metric() -> None:
    with pytest.raises(ValueError, match="preregistered"):
        EarlyStopping(metric="validation/core/view/macro_auprc")
