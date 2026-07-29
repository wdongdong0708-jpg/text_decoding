from __future__ import annotations

import torch

from eeg_keyword_decoding.losses import ot_context_loss
from eeg_keyword_decoding.models import TextProjectionOutput
from eeg_keyword_decoding.ot import (
    ContextOTOutput,
    CosineCostOutput,
    SinkhornOutput,
    TransportPoolOutput,
)


def _alignment() -> ContextOTOutput:
    plan = torch.tensor(
        [[[0.4, 0.1], [0.1, 0.4]], [[0.5, 0.0], [0.0, 0.5]]],
        requires_grad=True,
    )
    cost = torch.tensor(
        [[[0.2, 0.8], [0.7, 0.1]], [[0.3, 2.0], [2.0, 0.5]]],
        requires_grad=True,
    )
    per_sample = (plan * cost).sum(dim=(1, 2))
    return ContextOTOutput(
        text=TextProjectionOutput(
            sequence=torch.zeros(2, 2, 3),
            mask=torch.ones(2, 2, dtype=torch.bool),
            scalar_mix_weights=None,
        ),
        cost=CosineCostOutput(
            cost=cost,
            valid_pair_mask=torch.ones(2, 2, 2, dtype=torch.bool),
        ),
        sinkhorn=SinkhornOutput(
            plan=plan,
            transport_cost=per_sample,
            entropy=torch.tensor([1.0, 2.0]),
            row_marginal=plan.sum(dim=2),
            column_marginal=plan.sum(dim=1),
            row_error=torch.zeros(2),
            column_error=torch.zeros(2),
            iterations=10,
            epsilon=0.1,
        ),
        word_conditioned_eeg=TransportPoolOutput(
            sequence=torch.zeros(2, 2, 3),
            mask=torch.ones(2, 2, dtype=torch.bool),
            column_mass=plan.sum(dim=1),
        ),
    )


def test_ot_context_is_batch_mean_of_per_sample_expected_cost():
    alignment = _alignment()
    output = ot_context_loss(alignment)
    expected = (alignment.sinkhorn.plan * alignment.cost.cost).sum(
        dim=(1, 2)
    )
    assert torch.allclose(output.per_sample_loss, expected)
    assert torch.allclose(output.loss, expected.mean())
    assert torch.equal(output.entropy, alignment.sinkhorn.entropy)


def test_ot_context_backpropagates_through_plan_and_cost():
    alignment = _alignment()
    output = ot_context_loss(alignment)
    output.loss.backward()
    assert alignment.sinkhorn.plan.grad is not None
    assert alignment.cost.cost.grad is not None
    assert bool(torch.isfinite(alignment.sinkhorn.plan.grad).all())
    assert bool(torch.isfinite(alignment.cost.cost.grad).all())
