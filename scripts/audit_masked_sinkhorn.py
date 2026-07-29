from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_keyword_decoding.ot import (  # noqa: E402
    MaskedBalancedSinkhorn,
    MaskedBalancedSinkhornConfig,
)


DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs" / "ot" / "masked_balanced_sinkhorn_v1.yaml"
)


def _prefix_mask(lengths: list[int], maximum: int) -> torch.Tensor:
    return (
        torch.arange(maximum).unsqueeze(0)
        < torch.tensor(lengths, dtype=torch.int64).unsqueeze(1)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit balanced log-domain Sinkhorn invariants."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = MaskedBalancedSinkhornConfig.from_yaml(args.config.resolve())
    sinkhorn = MaskedBalancedSinkhorn.from_config(config)

    torch.manual_seed(31)
    eeg_mask = _prefix_mask([7, 4], 7)
    word_mask = _prefix_mask([5, 2], 5)
    cost = torch.rand(2, 7, 5, requires_grad=True)
    output = sinkhorn(
        cost=cost,
        eeg_mask=eeg_mask,
        word_mask=word_mask,
    )
    output.transport_cost.mean().backward()
    assert cost.grad is not None

    base_cost = cost.detach()[:1, :7, :5]
    base = sinkhorn(
        cost=base_cost,
        eeg_mask=torch.ones(1, 7, dtype=torch.bool),
        word_mask=torch.ones(1, 5, dtype=torch.bool),
    )
    padded_cost = torch.rand(1, 11, 9) * 100_000
    padded_cost[:, :7, :5] = base_cost
    padded = sinkhorn(
        cost=padded_cost,
        eeg_mask=_prefix_mask([7], 11),
        word_mask=_prefix_mask([5], 9),
    )
    plan_padding_error = float(
        (
            padded.plan[:, :7, :5]
            - base.plan
        ).abs().max()
    )
    cost_padding_error = float(
        (padded.transport_cost - base.transport_cost).abs().max()
    )
    pair_mask = eeg_mask.unsqueeze(2) & word_mask.unsqueeze(1)

    equal = sinkhorn(
        cost=torch.ones(1, 3, 2),
        eeg_mask=torch.ones(1, 3, dtype=torch.bool),
        word_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    product_error = float(
        (equal.plan - torch.full_like(equal.plan, 1.0 / 6.0))
        .abs()
        .max()
    )
    result: dict[str, Any] = {
        "config_path": str(args.config.resolve()),
        "config": {
            "type": config.transport_type,
            "cost": config.cost,
            "epsilon": config.epsilon,
            "iterations": config.iterations,
            "internal_dtype": config.internal_dtype,
            "time_marginal": config.time_marginal,
            "word_marginal": config.word_marginal,
            "position_cost": config.position_cost,
            "order_constraint": config.order_constraint,
            "marginal_audit_tolerance": (
                config.marginal_audit_tolerance
            ),
        },
        "parameter_count": sum(
            parameter.numel() for parameter in sinkhorn.parameters()
        ),
        "plan_dtype": str(output.plan.dtype),
        "plan_total_mass": output.plan.sum(dim=(1, 2)).tolist(),
        "maximum_row_error": float(output.row_error.detach().max()),
        "maximum_column_error": float(
            output.column_error.detach().max()
        ),
        "padding_plan_max_abs_value": float(
            output.plan.detach()[~pair_mask].abs().max()
        ),
        "padding_invariance": {
            "valid_plan_max_abs_error": plan_padding_error,
            "transport_cost_abs_error": cost_padding_error,
        },
        "equal_cost_product_marginal_max_abs_error": product_error,
        "transport_cost_range": [
            float(output.transport_cost.detach().min()),
            float(output.transport_cost.detach().max()),
        ],
        "entropy_range": [
            float(output.entropy.detach().min()),
            float(output.entropy.detach().max()),
        ],
        "finite_outputs": all(
            bool(torch.isfinite(value).all())
            for value in (
                output.plan,
                output.transport_cost,
                output.entropy,
                output.row_marginal,
                output.column_marginal,
            )
        ),
        "finite_gradients": bool(torch.isfinite(cost.grad).all()),
    }
    if result["maximum_row_error"] > config.marginal_audit_tolerance:
        raise ValueError("Row marginal audit tolerance exceeded")
    if result["maximum_column_error"] > config.marginal_audit_tolerance:
        raise ValueError("Column marginal audit tolerance exceeded")
    if plan_padding_error > 1e-6 or cost_padding_error > 1e-6:
        raise ValueError("Padding changed valid Sinkhorn output")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
