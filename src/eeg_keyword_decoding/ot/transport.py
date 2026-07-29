from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from .outputs import TransportPoolOutput


TRANSPORT_POOL_CONFIG_SCHEMA = "transport_pooling_v1"


@dataclass(frozen=True)
class TransportPoolingConfig:
    schema_version: str
    normalize_by_column_mass: bool
    eps: float

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TransportPoolingConfig:
        config = cls(
            schema_version=str(value["schema_version"]),
            normalize_by_column_mass=bool(
                value["normalize_by_column_mass"]
            ),
            eps=float(value["eps"]),
        )
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> TransportPoolingConfig:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Transport YAML root must be a mapping")
        return cls.from_dict(value)

    def validate(self) -> None:
        if self.schema_version != TRANSPORT_POOL_CONFIG_SCHEMA:
            raise ValueError(
                f"Unsupported transport pooling schema: "
                f"{self.schema_version!r}"
            )
        if not self.normalize_by_column_mass:
            raise ValueError("v1 requires column-mass normalization")
        if self.eps <= 0:
            raise ValueError("transport eps must be positive")


def transport_pool_eeg_to_words(
    *,
    eeg_sequence: torch.Tensor,
    plan: torch.Tensor,
    word_mask: torch.Tensor,
    eps: float = 1e-8,
) -> TransportPoolOutput:
    if eps <= 0:
        raise ValueError("eps must be positive")
    if eeg_sequence.ndim != 3:
        raise ValueError(
            "eeg_sequence must have shape [batch, time, feature]"
        )
    if plan.ndim != 3:
        raise ValueError("plan must have shape [batch, time, word]")
    if not eeg_sequence.is_floating_point() or not plan.is_floating_point():
        raise ValueError("eeg_sequence and plan must be floating point")
    batch_size, time_count, _ = eeg_sequence.shape
    if plan.shape[:2] != (batch_size, time_count):
        raise ValueError("plan time axes do not match eeg_sequence")
    if word_mask.shape != (batch_size, plan.shape[2]):
        raise ValueError("word_mask shape does not match plan")
    if word_mask.dtype != torch.bool:
        raise ValueError("word_mask must be torch.bool")
    if (
        eeg_sequence.device != plan.device
        or plan.device != word_mask.device
    ):
        raise ValueError("eeg_sequence, plan, and word_mask must share a device")
    if not bool(torch.isfinite(eeg_sequence).all()):
        raise ValueError("eeg_sequence contains NaN or Inf")
    if not bool(torch.isfinite(plan).all()):
        raise ValueError("plan contains NaN or Inf")
    if bool((plan < 0).any()):
        raise ValueError("plan must be non-negative")

    plan_fp32 = plan.float()
    column_mass = plan_fp32.sum(dim=1)
    if bool((column_mass[word_mask] <= 0).any()):
        raise ValueError("Every valid word must receive positive mass")
    numerator = torch.einsum(
        "btn,btd->bnd",
        plan_fp32,
        eeg_sequence.float(),
    )
    sequence = numerator / column_mass.clamp_min(eps).unsqueeze(-1)
    sequence = sequence.masked_fill(~word_mask.unsqueeze(-1), 0.0)
    column_mass = column_mass.masked_fill(~word_mask, 0.0)
    if not bool(torch.isfinite(sequence).all()):
        raise FloatingPointError("Transport pooling produced NaN or Inf")
    return TransportPoolOutput(
        sequence=sequence,
        mask=word_mask,
        column_mass=column_mass,
    )
