from eeg_keyword_decoding.training.synthetic import run_synthetic_tiny_overfit


def test_synthetic_tiny_overfit_decreases_loss_and_resumes_exactly() -> None:
    first = run_synthetic_tiny_overfit(steps=25, seed=42)
    second = run_synthetic_tiny_overfit(steps=25, seed=42)
    assert first.first_window_mean_loss == second.first_window_mean_loss
    assert first.final_window_mean_loss == second.final_window_mean_loss
    assert first.final_window_mean_loss < first.first_window_mean_loss
    assert first.final_prototype_accuracy >= first.initial_prototype_accuracy
    assert first.maximum_gradient > 0
    assert first.padding_loss_absolute_error < 1e-5
    assert first.resumed_next_loss_absolute_error == 0
    assert first.resumed_parameter_max_absolute_error == 0
