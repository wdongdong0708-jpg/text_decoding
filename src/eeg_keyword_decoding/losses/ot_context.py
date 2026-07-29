from __future__ import annotations

import torch

from eeg_keyword_decoding.ot import ContextOTOutput

from .outputs import OTContextLossOutput


def ot_context_loss(
    alignment: ContextOTOutput,
    *,
    mass_atol: float = 1e-3,
) -> OTContextLossOutput:
    """Mean per-sample expected transport cost.

    Entropy remains a diagnostic and is deliberately not added to this loss.
    The transport plan is not detached, so gradients pass through Sinkhorn.
    """

    if mass_atol <= 0:
        raise ValueError("mass_atol must be positive")
    plan = alignment.sinkhorn.plan
    cost = alignment.cost.cost.float()
    if plan.shape != cost.shape:
        raise ValueError("Sinkhorn plan and cosine cost shapes differ")
    if plan.ndim != 3 or plan.shape[0] <= 0:
        raise ValueError("Expected a non-empty [batch,time,word] plan")
    per_sample = (plan * cost).sum(dim=(1, 2))
    if not torch.allclose(
        per_sample,
        alignment.sinkhorn.transport_cost,
        atol=1e-6,
        rtol=1e-5,
    ):
        raise ValueError(
            "Sinkhorn transport_cost disagrees with plan-weighted cost"
        )
    plan_total_mass = plan.sum(dim=(1, 2))
    if not torch.allclose(
        plan_total_mass,
        torch.ones_like(plan_total_mass),
        atol=mass_atol,
        rtol=0.0,
    ):
        raise ValueError(
            "Balanced transport plan total mass differs from one: "
            f"{plan_total_mass.detach().cpu().tolist()}"
        )
    loss = per_sample.mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("OT-context loss is NaN or Inf")
    return OTContextLossOutput(
        loss=loss,
        per_sample_loss=per_sample,
        plan_total_mass=plan_total_mass,
        entropy=alignment.sinkhorn.entropy,
    )
