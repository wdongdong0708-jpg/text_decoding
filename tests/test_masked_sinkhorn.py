from __future__ import annotations

from pathlib import Path

import pytest
import torch

from eeg_keyword_decoding.ot import (
    MaskedBalancedSinkhorn,
    MaskedBalancedSinkhornConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def prefix_mask(lengths: list[int], maximum: int) -> torch.Tensor:
    return (
        torch.arange(maximum).unsqueeze(0)
        < torch.tensor(lengths).unsqueeze(1)
    )


def run_random_sinkhorn(
    *,
    time_lengths=(4, 7),
    word_lengths=(3, 2),
    epsilon=0.05,
    iterations=100,
):
    time_max = max(time_lengths)
    word_max = max(word_lengths)
    eeg_mask = prefix_mask(list(time_lengths), time_max)
    word_mask = prefix_mask(list(word_lengths), word_max)
    cost = torch.rand(len(time_lengths), time_max, word_max)
    return (
        MaskedBalancedSinkhorn(
            epsilon=epsilon,
            iterations=iterations,
        )(
            cost=cost,
            eeg_mask=eeg_mask,
            word_mask=word_mask,
        ),
        cost,
        eeg_mask,
        word_mask,
    )


def test_sinkhorn_plan_mass_marginals_and_invalid_entries():
    output, _, eeg_mask, word_mask = run_random_sinkhorn()
    torch.testing.assert_close(
        output.plan.sum(dim=(1, 2)),
        torch.ones(2),
        atol=1e-5,
        rtol=1e-5,
    )
    expected_rows = eeg_mask / eeg_mask.sum(dim=1, keepdim=True)
    expected_columns = word_mask / word_mask.sum(dim=1, keepdim=True)
    torch.testing.assert_close(
        output.row_marginal,
        expected_rows,
        atol=1e-4,
        rtol=1e-4,
    )
    torch.testing.assert_close(
        output.column_marginal,
        expected_columns,
        atol=1e-4,
        rtol=1e-4,
    )
    pair_mask = eeg_mask.unsqueeze(2) & word_mask.unsqueeze(1)
    assert torch.count_nonzero(output.plan[~pair_mask]) == 0
    assert output.plan.dtype == torch.float32
    assert output.row_error.max() < 1e-4
    assert output.column_error.max() < 1e-4


@pytest.mark.parametrize(
    ("time_lengths", "word_lengths"),
    [((2, 8), (1, 4)), ((1,), (1,)), ((3, 4, 9), (2, 1, 5))],
)
def test_sinkhorn_supports_variable_short_sequences(
    time_lengths,
    word_lengths,
):
    output, *_ = run_random_sinkhorn(
        time_lengths=time_lengths,
        word_lengths=word_lengths,
        iterations=150,
    )
    assert torch.isfinite(output.plan).all()
    assert torch.isfinite(output.transport_cost).all()
    assert torch.isfinite(output.entropy).all()


def test_equal_cost_plan_is_product_of_uniform_marginals():
    eeg_mask = prefix_mask([3], 3)
    word_mask = prefix_mask([2], 2)
    output = MaskedBalancedSinkhorn(iterations=10)(
        cost=torch.ones(1, 3, 2),
        eeg_mask=eeg_mask,
        word_mask=word_mask,
    )
    torch.testing.assert_close(
        output.plan,
        torch.full((1, 3, 2), 1.0 / 6.0),
        atol=1e-6,
        rtol=1e-6,
    )


def test_sinkhorn_rejects_empty_nonprefix_nonfinite_and_invalid_parameters():
    with pytest.raises(ValueError, match="epsilon"):
        MaskedBalancedSinkhorn(epsilon=0)
    with pytest.raises(ValueError, match="iterations"):
        MaskedBalancedSinkhorn(iterations=0)
    module = MaskedBalancedSinkhorn()
    with pytest.raises(ValueError, match="empty sequence"):
        module(
            cost=torch.ones(1, 2, 2),
            eeg_mask=torch.zeros(1, 2, dtype=torch.bool),
            word_mask=torch.ones(1, 2, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="empty sequence"):
        module(
            cost=torch.ones(1, 2, 2),
            eeg_mask=torch.ones(1, 2, dtype=torch.bool),
            word_mask=torch.zeros(1, 2, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="contiguous true prefix"):
        module(
            cost=torch.ones(1, 3, 2),
            eeg_mask=torch.tensor([[True, False, True]]),
            word_mask=torch.ones(1, 2, dtype=torch.bool),
        )
    bad = torch.ones(1, 2, 2)
    bad[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        module(
            cost=bad,
            eeg_mask=torch.ones(1, 2, dtype=torch.bool),
            word_mask=torch.ones(1, 2, dtype=torch.bool),
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_external_cost_produces_fp32_finite_plan(dtype):
    output = MaskedBalancedSinkhorn(epsilon=0.01, iterations=150)(
        cost=(torch.rand(2, 6, 4) * 100).to(dtype),
        eeg_mask=prefix_mask([4, 6], 6),
        word_mask=prefix_mask([3, 4], 4),
    )
    assert output.plan.dtype == torch.float32
    assert torch.isfinite(output.plan).all()


def test_small_epsilon_and_large_cost_differences_remain_finite():
    cost = torch.tensor(
        [[[0.0, 1000.0], [1000.0, 0.0], [500.0, 500.0]]]
    )
    output = MaskedBalancedSinkhorn(
        epsilon=1e-3,
        iterations=200,
    )(
        cost=cost,
        eeg_mask=torch.ones(1, 3, dtype=torch.bool),
        word_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    assert torch.isfinite(output.plan).all()
    assert torch.isfinite(output.transport_cost).all()


def test_sinkhorn_padding_invariance_and_single_vs_batch_equivalence():
    torch.manual_seed(9)
    cost = torch.rand(1, 3, 2)
    module = MaskedBalancedSinkhorn(iterations=100)
    baseline = module(
        cost=cost,
        eeg_mask=torch.ones(1, 3, dtype=torch.bool),
        word_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    padded_cost = torch.rand(1, 8, 6) * 100_000
    padded_cost[:, :3, :2] = cost
    observed = module(
        cost=padded_cost,
        eeg_mask=prefix_mask([3], 8),
        word_mask=prefix_mask([2], 6),
    )
    torch.testing.assert_close(
        observed.plan[:, :3, :2],
        baseline.plan,
        atol=1e-7,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        observed.transport_cost,
        baseline.transport_cost,
        atol=1e-7,
        rtol=1e-6,
    )

    other_cost = torch.rand(1, 8, 6)
    batch_cost = torch.cat([padded_cost, other_cost], dim=0)
    batch = module(
        cost=batch_cost,
        eeg_mask=prefix_mask([3, 8], 8),
        word_mask=prefix_mask([2, 6], 6),
    )
    torch.testing.assert_close(
        batch.plan[0, :3, :2],
        baseline.plan[0],
        atol=1e-7,
        rtol=1e-6,
    )


def test_sinkhorn_time_word_and_batch_permutation_equivariance():
    torch.manual_seed(10)
    cost = torch.rand(2, 4, 3)
    module = MaskedBalancedSinkhorn(iterations=100)
    mask_t = torch.ones(2, 4, dtype=torch.bool)
    mask_w = torch.ones(2, 3, dtype=torch.bool)
    expected = module(cost=cost, eeg_mask=mask_t, word_mask=mask_w)
    time_perm = torch.tensor([2, 0, 3, 1])
    word_perm = torch.tensor([1, 2, 0])
    batch_perm = torch.tensor([1, 0])
    observed = module(
        cost=cost[
            batch_perm
        ][:, time_perm][:, :, word_perm],
        eeg_mask=mask_t[batch_perm][:, time_perm],
        word_mask=mask_w[batch_perm][:, word_perm],
    )
    inverse_batch = torch.argsort(batch_perm)
    inverse_time = torch.argsort(time_perm)
    inverse_word = torch.argsort(word_perm)
    restored = observed.plan[inverse_batch][
        :, inverse_time
    ][:, :, inverse_word]
    torch.testing.assert_close(restored, expected.plan, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(
        observed.transport_cost[inverse_batch],
        expected.transport_cost,
        atol=1e-6,
        rtol=1e-5,
    )


def test_transport_cost_backpropagates_finite_gradients():
    cost = torch.rand(2, 6, 4, requires_grad=True)
    output = MaskedBalancedSinkhorn(iterations=50)(
        cost=cost,
        eeg_mask=prefix_mask([4, 6], 6),
        word_mask=prefix_mask([3, 4], 4),
    )
    output.transport_cost.mean().backward()
    assert cost.grad is not None
    assert torch.isfinite(cost.grad).all()
    pair_mask = (
        prefix_mask([4, 6], 6).unsqueeze(2)
        & prefix_mask([3, 4], 4).unsqueeze(1)
    )
    assert torch.count_nonzero(cost.grad[~pair_mask]) == 0


def test_sinkhorn_config_file_freezes_balanced_fp32_contract():
    config = MaskedBalancedSinkhornConfig.from_yaml(
        PROJECT_ROOT
        / "configs"
        / "ot"
        / "masked_balanced_sinkhorn_v1.yaml"
    )
    assert config.epsilon == 0.05
    assert config.iterations == 50
    assert config.internal_dtype == "float32"
    assert not config.position_cost
    assert not config.order_constraint


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_autocast_sinkhorn_stays_fp32_and_finite():
    device = torch.device("cuda")
    cost = torch.rand(2, 6, 4, device=device, requires_grad=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = MaskedBalancedSinkhorn(iterations=50)(
            cost=cost,
            eeg_mask=prefix_mask([4, 6], 6).to(device),
            word_mask=prefix_mask([3, 4], 4).to(device),
        )
        loss = output.transport_cost.mean()
    loss.backward()
    assert output.plan.dtype == torch.float32
    assert torch.isfinite(output.plan).all()
    assert cost.grad is not None and torch.isfinite(cost.grad).all()
