from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from eeg_keyword_decoding.ot import ContextOTOutput
from eeg_keyword_decoding.prototypes import PrototypeBank

from .context_token import context_token_info_nce
from .ot_context import ot_context_loss
from .outputs import (
    ContextOTLossOutput,
    ContextTokenLossOutput,
    PrototypeLossOutput,
)
from .prototype import prototype_classification_loss


THREE_LOSS_CONFIG_SCHEMA = "context_ot_three_loss_v1"


@dataclass(frozen=True)
class ThreeLossConfig:
    schema_version: str
    ot_weight: float
    token_weight: float
    token_temperature: float
    token_symmetric: bool
    token_reduction: str
    prototype_weight: float
    prototype_temperature: float
    prototype_min_train_group_df: int
    prototype_reduction: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ThreeLossConfig:
        losses = dict(value["loss"])
        ot = dict(losses["ot_context"])
        token = dict(losses["context_token"])
        prototype = dict(losses["prototype"])
        config = cls(
            schema_version=str(value["schema_version"]),
            ot_weight=float(ot["weight"]),
            token_weight=float(token["weight"]),
            token_temperature=float(token["temperature"]),
            token_symmetric=bool(token["symmetric"]),
            token_reduction=str(token["reduction"]),
            prototype_weight=float(prototype["weight"]),
            prototype_temperature=float(prototype["temperature"]),
            prototype_min_train_group_df=int(
                prototype["min_train_group_df"]
            ),
            prototype_reduction=str(prototype["reduction"]),
        )
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> ThreeLossConfig:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Three-loss YAML root must be a mapping")
        return cls.from_dict(value)

    def validate(self) -> None:
        if self.schema_version != THREE_LOSS_CONFIG_SCHEMA:
            raise ValueError(
                f"Unsupported three-loss schema: {self.schema_version!r}"
            )
        weights = (
            self.ot_weight,
            self.token_weight,
            self.prototype_weight,
        )
        if any(value < 0 for value in weights):
            raise ValueError("All loss weights must be non-negative")
        if not any(value > 0 for value in weights):
            raise ValueError("At least one loss weight must be positive")
        if self.token_temperature <= 0:
            raise ValueError("Token temperature must be positive")
        if self.prototype_temperature <= 0:
            raise ValueError("Prototype temperature must be positive")
        if self.token_reduction != "sample_balanced":
            raise ValueError("Default token reduction must be sample_balanced")
        if self.prototype_reduction != "sample_balanced":
            raise ValueError(
                "Default prototype reduction must be sample_balanced"
            )
        if self.prototype_min_train_group_df <= 0:
            raise ValueError("Prototype group DF threshold must be positive")

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "loss": {
                "ot_context": {"weight": self.ot_weight},
                "context_token": {
                    "weight": self.token_weight,
                    "temperature": self.token_temperature,
                    "symmetric": self.token_symmetric,
                    "reduction": self.token_reduction,
                },
                "prototype": {
                    "weight": self.prototype_weight,
                    "temperature": self.prototype_temperature,
                    "min_train_group_df": (
                        self.prototype_min_train_group_df
                    ),
                    "reduction": self.prototype_reduction,
                },
            },
        }


def _zero_token_output(
    connected: torch.Tensor,
    *,
    batch_size: int,
) -> ContextTokenLossOutput:
    zero = connected.sum() * 0.0
    empty = torch.empty(0, dtype=torch.float32, device=connected.device)
    sample = torch.zeros(
        batch_size,
        dtype=torch.float32,
        device=connected.device,
    )
    counts = torch.zeros(
        batch_size,
        dtype=torch.int64,
        device=connected.device,
    )
    return ContextTokenLossOutput(
        loss=zero,
        eeg_to_text_loss=zero,
        text_to_eeg_loss=zero,
        eeg_to_text_query_loss=empty,
        text_to_eeg_query_loss=empty,
        eeg_to_text_sample_loss=sample,
        text_to_eeg_sample_loss=sample,
        valid_words_per_sample=counts,
        positive_counts=torch.empty(
            0,
            dtype=torch.int64,
            device=connected.device,
        ),
        negative_counts=torch.empty(
            0,
            dtype=torch.int64,
            device=connected.device,
        ),
        valid_query_count=0,
        contributing_query_count=0,
        false_negative_mask_count=0,
    )


def _zero_prototype_output(
    connected: torch.Tensor,
    *,
    batch_size: int,
    word_count: int,
    available_count: int,
) -> PrototypeLossOutput:
    zero = connected.sum() * 0.0
    return PrototypeLossOutput(
        loss=zero,
        query_loss=torch.zeros(
            (batch_size, word_count),
            dtype=torch.float32,
            device=connected.device,
        ),
        sample_loss=torch.zeros(
            batch_size,
            dtype=torch.float32,
            device=connected.device,
        ),
        valid_words_per_sample=torch.zeros(
            batch_size,
            dtype=torch.int64,
            device=connected.device,
        ),
        valid_word_count=0,
        valid_sample_count=0,
        ignored_non_master_count=0,
        ignored_unavailable_target_count=0,
        available_prototype_count=available_count,
    )


class ContextOTThreeLoss(nn.Module):
    def __init__(self, config: ThreeLossConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

    def forward(
        self,
        *,
        alignment: ContextOTOutput,
        word_mask: torch.Tensor,
        context_token_group_indices: torch.Tensor,
        surface_type_indices: torch.Tensor,
        sentence_group_indices: torch.Tensor,
        word_keyword_indices: torch.Tensor,
        prototype_bank: PrototypeBank | None,
    ) -> ContextOTLossOutput:
        batch_size, word_count = word_mask.shape
        ot_output = ot_context_loss(alignment)
        connected = alignment.word_conditioned_eeg.sequence

        if self.config.token_weight > 0:
            token_output = context_token_info_nce(
                word_conditioned_eeg=connected,
                projected_context_words=alignment.text.sequence,
                word_mask=word_mask,
                context_token_group_indices=context_token_group_indices,
                surface_type_indices=surface_type_indices,
                sentence_group_indices=sentence_group_indices,
                temperature=self.config.token_temperature,
                symmetric=self.config.token_symmetric,
            )
        else:
            token_output = _zero_token_output(
                connected,
                batch_size=batch_size,
            )

        if self.config.prototype_weight > 0:
            if prototype_bank is None:
                raise ValueError(
                    "Enabled prototype loss requires a PrototypeBank"
                )
            if (
                prototype_bank.metadata.get("min_train_group_df")
                != self.config.prototype_min_train_group_df
            ):
                raise ValueError(
                    "Prototype bank group DF threshold disagrees with loss "
                    "configuration"
                )
            prototype_output = prototype_classification_loss(
                word_conditioned_eeg=connected,
                word_keyword_indices=word_keyword_indices,
                word_mask=word_mask,
                prototype_bank=prototype_bank,
                temperature=self.config.prototype_temperature,
            )
        else:
            prototype_output = _zero_prototype_output(
                connected,
                batch_size=batch_size,
                word_count=word_count,
                available_count=(
                    prototype_bank.available_count
                    if prototype_bank is not None
                    else 0
                ),
            )

        weighted_ot = ot_output.loss * self.config.ot_weight
        weighted_token = token_output.loss * self.config.token_weight
        weighted_prototype = (
            prototype_output.loss * self.config.prototype_weight
        )
        total = weighted_ot + weighted_token + weighted_prototype
        values = (
            total,
            ot_output.loss,
            token_output.loss,
            prototype_output.loss,
            weighted_ot,
            weighted_token,
            weighted_prototype,
        )
        finite = all(bool(torch.isfinite(value)) for value in values)
        if not finite:
            raise FloatingPointError("Three-loss output contains NaN or Inf")
        scalar_mix = alignment.text.scalar_mix_weights
        diagnostics = {
            "batch_size": batch_size,
            "eeg_valid_lengths": (
                (alignment.sinkhorn.row_marginal > 0)
                .sum(dim=1)
                .detach()
                .cpu()
                .tolist()
            ),
            "word_valid_lengths": word_mask.sum(dim=1).cpu().tolist(),
            "transport_cost_mean": float(
                ot_output.per_sample_loss.detach().mean().cpu()
            ),
            "transport_cost_min": float(
                ot_output.per_sample_loss.detach().min().cpu()
            ),
            "transport_cost_max": float(
                ot_output.per_sample_loss.detach().max().cpu()
            ),
            "plan_row_error_max": float(
                alignment.sinkhorn.row_error.detach().max().cpu()
            ),
            "plan_column_error_max": float(
                alignment.sinkhorn.column_error.detach().max().cpu()
            ),
            "token_valid_query_count": token_output.valid_query_count,
            "token_contributing_query_count": (
                token_output.contributing_query_count
            ),
            "token_mean_positive_count": (
                float(token_output.positive_counts.float().mean().cpu())
                if token_output.positive_counts.numel()
                else 0.0
            ),
            "token_false_negative_mask_count": (
                token_output.false_negative_mask_count
            ),
            "prototype_valid_word_count": (
                prototype_output.valid_word_count
            ),
            "prototype_valid_sample_count": (
                prototype_output.valid_sample_count
            ),
            "available_prototype_count": (
                prototype_output.available_prototype_count
            ),
            "scalar_mix_weights": (
                scalar_mix.detach().cpu().tolist()
                if scalar_mix is not None
                else None
            ),
            "loss_unweighted": {
                "ot_context": float(ot_output.loss.detach().cpu()),
                "context_token": float(token_output.loss.detach().cpu()),
                "prototype": float(prototype_output.loss.detach().cpu()),
            },
            "loss_weighted": {
                "ot_context": float(weighted_ot.detach().cpu()),
                "context_token": float(weighted_token.detach().cpu()),
                "prototype": float(weighted_prototype.detach().cpu()),
            },
            "all_finite": finite,
            "zero_weight_skips": {
                "ot_context": self.config.ot_weight == 0,
                "context_token": self.config.token_weight == 0,
                "prototype": self.config.prototype_weight == 0,
            },
        }
        return ContextOTLossOutput(
            total=total,
            ot_context=ot_output.loss,
            context_token=token_output.loss,
            prototype=prototype_output.loss,
            weighted_ot=weighted_ot,
            weighted_token=weighted_token,
            weighted_prototype=weighted_prototype,
            ot_output=ot_output,
            token_output=token_output,
            prototype_output=prototype_output,
            diagnostics=diagnostics,
        )
