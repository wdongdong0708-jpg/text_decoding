from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from .outputs import SinkhornOutput


SINKHORN_CONFIG_SCHEMA = "masked_balanced_sinkhorn_v1"


@dataclass(frozen=True)
class MaskedBalancedSinkhornConfig:
    schema_version: str
    transport_type: str
    cost: str
    epsilon: float
    iterations: int
    marginal_audit_tolerance: float
    internal_dtype: str
    time_marginal: str
    word_marginal: str
    position_cost: bool
    order_constraint: bool

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
    ) -> MaskedBalancedSinkhornConfig:
        config = cls(
            schema_version=str(value["schema_version"]),
            transport_type=str(value["type"]),
            cost=str(value["cost"]),
            epsilon=float(value["epsilon"]),
            iterations=int(value["iterations"]),
            marginal_audit_tolerance=float(
                value["marginal_audit_tolerance"]
            ),
            internal_dtype=str(value["internal_dtype"]),
            time_marginal=str(value["time_marginal"]),
            word_marginal=str(value["word_marginal"]),
            position_cost=bool(value["position_cost"]),
            order_constraint=bool(value["order_constraint"]),
        )
        config.validate()
        return config

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
    ) -> MaskedBalancedSinkhornConfig:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Sinkhorn YAML root must be a mapping")
        return cls.from_dict(value)

    def validate(self) -> None:
        if self.schema_version != SINKHORN_CONFIG_SCHEMA:
            raise ValueError(
                f"Unsupported Sinkhorn schema: {self.schema_version!r}"
            )
        if self.transport_type != "balanced":
            raise ValueError("v1 supports only balanced OT")
        if self.cost != "cosine":
            raise ValueError("v1 supports only cosine cost")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.marginal_audit_tolerance <= 0:
            raise ValueError("marginal_audit_tolerance must be positive")
        if self.internal_dtype != "float32":
            raise ValueError("Sinkhorn internal_dtype must be float32")
        if self.time_marginal != "uniform":
            raise ValueError("v1 requires a uniform time marginal")
        if self.word_marginal != "uniform":
            raise ValueError("v1 requires a uniform word marginal")
        if self.position_cost:
            raise ValueError("v1 forbids a position cost")
        if self.order_constraint:
            raise ValueError("v1 forbids an order constraint")


def _prefix_lengths(mask: torch.Tensor, *, name: str) -> torch.Tensor:
    if mask.ndim != 2:
        raise ValueError(f"{name} must have shape [batch, position]")
    if mask.dtype != torch.bool:
        raise ValueError(f"{name} must be torch.bool")
    if mask.shape[0] <= 0 or mask.shape[1] <= 0:
        raise ValueError(f"{name} has an empty axis")
    lengths = mask.sum(dim=1, dtype=torch.int64)
    if bool((lengths == 0).any()):
        raise ValueError(f"{name} contains an empty sequence")
    expected = (
        torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
        < lengths.unsqueeze(1)
    )
    if not torch.equal(mask, expected):
        raise ValueError(
            f"{name} must be a contiguous true prefix followed by padding"
        )
    return lengths


def _autocast_disabled(device_type: str):
    if device_type in {"cpu", "cuda"}:
        return torch.autocast(device_type=device_type, enabled=False)
    return nullcontext()


class MaskedBalancedSinkhorn(nn.Module):
    """Batched differentiable balanced Sinkhorn in the log domain."""

    def __init__(
        self,
        *,
        epsilon: float = 0.05,
        iterations: int = 50,
    ) -> None:
        super().__init__()
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        self.epsilon = float(epsilon)
        self.iterations = int(iterations)

    @classmethod
    def from_config(
        cls,
        config: MaskedBalancedSinkhornConfig,
    ) -> MaskedBalancedSinkhorn:
        config.validate()
        return cls(
            epsilon=config.epsilon,
            iterations=config.iterations,
        )

    def forward(
        self,
        *,
        cost: torch.Tensor,
        eeg_mask: torch.Tensor,
        word_mask: torch.Tensor,
        valid_pair_mask: torch.Tensor | None = None,
    ) -> SinkhornOutput:
        if cost.ndim != 3:
            raise ValueError(
                f"cost must have shape [batch, time, word], got "
                f"{tuple(cost.shape)}"
            )
        if not cost.is_floating_point():
            raise ValueError("cost must be floating point")
        batch_size, time_count, word_count = cost.shape
        if eeg_mask.shape != (batch_size, time_count):
            raise ValueError("eeg_mask shape does not match cost")
        if word_mask.shape != (batch_size, word_count):
            raise ValueError("word_mask shape does not match cost")
        if (
            cost.device != eeg_mask.device
            or cost.device != word_mask.device
        ):
            raise ValueError("cost and masks must share a device")
        if not bool(torch.isfinite(cost).all()):
            raise ValueError("cost contains NaN or Inf")
        time_lengths = _prefix_lengths(eeg_mask, name="eeg_mask")
        word_lengths = _prefix_lengths(word_mask, name="word_mask")
        expected_pair_mask = eeg_mask.unsqueeze(2) & word_mask.unsqueeze(1)
        if valid_pair_mask is not None:
            if valid_pair_mask.shape != cost.shape:
                raise ValueError("valid_pair_mask shape does not match cost")
            if valid_pair_mask.dtype != torch.bool:
                raise ValueError("valid_pair_mask must be torch.bool")
            if valid_pair_mask.device != cost.device:
                raise ValueError("valid_pair_mask must share the cost device")
            if not torch.equal(valid_pair_mask, expected_pair_mask):
                raise ValueError(
                    "valid_pair_mask disagrees with EEG/word masks"
                )
        pair_mask = expected_pair_mask

        with _autocast_disabled(cost.device.type):
            cost_fp32 = cost.float()
            time_mass = (
                eeg_mask.float()
                / time_lengths.float().unsqueeze(1)
            )
            word_mass = (
                word_mask.float()
                / word_lengths.float().unsqueeze(1)
            )
            log_time_mass = torch.where(
                eeg_mask,
                torch.log(time_mass.clamp_min(torch.finfo(torch.float32).tiny)),
                torch.zeros_like(time_mass),
            )
            log_word_mass = torch.where(
                word_mask,
                torch.log(word_mass.clamp_min(torch.finfo(torch.float32).tiny)),
                torch.zeros_like(word_mass),
            )
            log_kernel = (-cost_fp32 / self.epsilon).masked_fill(
                ~pair_mask,
                -torch.inf,
            )
            log_u = torch.zeros_like(time_mass)
            log_v = torch.zeros_like(word_mass)

            for _ in range(self.iterations):
                row_values = log_kernel + log_v.unsqueeze(1)
                row_values = torch.where(
                    eeg_mask.unsqueeze(2),
                    row_values,
                    torch.zeros_like(row_values),
                )
                row_logsum = torch.logsumexp(
                    row_values,
                    dim=2,
                )
                log_u = torch.where(
                    eeg_mask,
                    log_time_mass - row_logsum,
                    torch.zeros_like(log_u),
                )
                column_values = log_kernel + log_u.unsqueeze(2)
                column_values = torch.where(
                    word_mask.unsqueeze(1),
                    column_values,
                    torch.zeros_like(column_values),
                )
                column_logsum = torch.logsumexp(
                    column_values,
                    dim=1,
                )
                log_v = torch.where(
                    word_mask,
                    log_word_mass - column_logsum,
                    torch.zeros_like(log_v),
                )

            log_plan = (
                log_kernel
                + log_u.unsqueeze(2)
                + log_v.unsqueeze(1)
            )
            plan = torch.exp(log_plan).masked_fill(~pair_mask, 0.0)
            row_marginal = plan.sum(dim=2)
            column_marginal = plan.sum(dim=1)
            row_error = (row_marginal - time_mass).abs().amax(dim=1)
            column_error = (
                column_marginal - word_mass
            ).abs().amax(dim=1)
            transport_cost = (plan * cost_fp32).sum(dim=(1, 2))
            safe_plan = plan.clamp_min(torch.finfo(torch.float32).tiny)
            entropy = -(plan * safe_plan.log()).sum(dim=(1, 2))

        outputs = (
            plan,
            row_marginal,
            column_marginal,
            row_error,
            column_error,
            transport_cost,
            entropy,
        )
        if any(not bool(torch.isfinite(value).all()) for value in outputs):
            raise FloatingPointError("Sinkhorn produced NaN or Inf")
        return SinkhornOutput(
            plan=plan,
            transport_cost=transport_cost,
            entropy=entropy,
            row_marginal=row_marginal,
            column_marginal=column_marginal,
            row_error=row_error,
            column_error=column_error,
            iterations=self.iterations,
            epsilon=self.epsilon,
        )
