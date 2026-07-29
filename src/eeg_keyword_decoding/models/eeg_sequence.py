from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from .masked_ops import (
    lengths_from_prefix_mask,
    mask_channel_first,
    mask_time_first,
)
from .subject_adapter import SubjectFiLM
from .temporal_blocks import (
    MaskedDilatedResidualBlock,
    MaskedTemporalDownsample,
    TimewiseLayerNorm,
)


EEG_SEQUENCE_CONFIG_SCHEMA = "eeg_sequence_conv_v1"


@dataclass(frozen=True)
class SubjectAdapterConfig:
    enabled: bool
    adapter_type: str
    num_subjects: int
    gamma_initialization: float = 1.0
    beta_initialization: float = 0.0


@dataclass(frozen=True)
class EEGSequenceEncoderConfig:
    schema_version: str
    input_channels: int
    hidden_dim: int
    output_dim: int
    total_stride: int
    minimum_input_length: int
    stem_kernel_size: int
    stem_stride: int
    residual_kernel_size: int
    dilations: tuple[int, ...]
    downsample_kernel_size: int
    downsample_stride: int
    dropout: float
    normalization: str
    activation: str
    convolution_mode: str
    parameter_initialization: str
    mask_rule: str
    subject_adapter: SubjectAdapterConfig

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EEGSequenceEncoderConfig:
        adapter = dict(value["subject_adapter"])
        stem = dict(value["stem"])
        residual = dict(value["residual_blocks"])
        downsample = dict(value["downsample"])
        dilations = tuple(int(item) for item in residual["dilations"])
        if int(residual["count"]) != len(dilations):
            raise ValueError(
                "residual_blocks.count must equal the number of dilations"
            )
        config = cls(
            schema_version=str(value["schema_version"]),
            input_channels=int(value["input_channels"]),
            hidden_dim=int(value["hidden_dim"]),
            output_dim=int(value["output_dim"]),
            total_stride=int(value["total_stride"]),
            minimum_input_length=int(value["minimum_input_length"]),
            stem_kernel_size=int(stem["kernel_size"]),
            stem_stride=int(stem["stride"]),
            residual_kernel_size=int(residual["kernel_size"]),
            dilations=dilations,
            downsample_kernel_size=int(downsample["kernel_size"]),
            downsample_stride=int(downsample["stride"]),
            dropout=float(value["dropout"]),
            normalization=str(value["normalization"]),
            activation=str(value["activation"]),
            convolution_mode=str(value["convolution_mode"]),
            parameter_initialization=str(value["parameter_initialization"]),
            mask_rule=str(value["mask_rule"]),
            subject_adapter=SubjectAdapterConfig(
                enabled=bool(adapter["enabled"]),
                adapter_type=str(adapter["type"]),
                num_subjects=int(adapter["num_subjects"]),
                gamma_initialization=float(
                    adapter["gamma_initialization"]
                ),
                beta_initialization=float(
                    adapter["beta_initialization"]
                ),
            ),
        )
        config.validate()
        return config

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
    ) -> EEGSequenceEncoderConfig:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("EEG model YAML root must be a mapping")
        return cls.from_dict(value)

    def validate(self) -> None:
        if self.schema_version != EEG_SEQUENCE_CONFIG_SCHEMA:
            raise ValueError(
                f"Unsupported EEG sequence schema: {self.schema_version!r}"
            )
        for name, value in (
            ("input_channels", self.input_channels),
            ("hidden_dim", self.hidden_dim),
            ("output_dim", self.output_dim),
            ("minimum_input_length", self.minimum_input_length),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.total_stride != self.stem_stride * self.downsample_stride:
            raise ValueError(
                "total_stride must equal stem_stride * downsample_stride"
            )
        if self.total_stride != 4:
            raise ValueError("v1 freezes total_stride at 4")
        for name, kernel in (
            ("stem_kernel_size", self.stem_kernel_size),
            ("residual_kernel_size", self.residual_kernel_size),
            ("downsample_kernel_size", self.downsample_kernel_size),
        ):
            if kernel <= 0 or kernel % 2 == 0:
                raise ValueError(f"{name} must be a positive odd integer")
        if not self.dilations or any(value <= 0 for value in self.dilations):
            raise ValueError("dilations must be a non-empty positive sequence")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if self.normalization != "timewise_layer_norm":
            raise ValueError(
                "v1 requires timewise_layer_norm; BatchNorm and ordinary "
                "GroupNorm are forbidden"
            )
        if self.activation not in {"gelu", "silu"}:
            raise ValueError("activation must be gelu or silu")
        if self.convolution_mode != "symmetric_noncausal":
            raise ValueError("v1 requires symmetric_noncausal convolution")
        if self.parameter_initialization != "pytorch_default":
            raise ValueError("Unsupported parameter initialization")
        if self.mask_rule != "zero_after_every_operation":
            raise ValueError("Unsupported mask rule")
        if self.subject_adapter.adapter_type != "film":
            raise ValueError("Only the FiLM subject adapter is supported")
        if self.subject_adapter.num_subjects <= 0:
            raise ValueError("subject adapter num_subjects must be positive")
        if self.subject_adapter.gamma_initialization != 1.0:
            raise ValueError("FiLM gamma initialization must be 1.0")
        if self.subject_adapter.beta_initialization != 0.0:
            raise ValueError("FiLM beta initialization must be 0.0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "input_channels": self.input_channels,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "total_stride": self.total_stride,
            "minimum_input_length": self.minimum_input_length,
            "stem": {
                "kernel_size": self.stem_kernel_size,
                "stride": self.stem_stride,
            },
            "residual_blocks": {
                "kernel_size": self.residual_kernel_size,
                "dilations": list(self.dilations),
                "count": len(self.dilations),
            },
            "downsample": {
                "kernel_size": self.downsample_kernel_size,
                "stride": self.downsample_stride,
            },
            "dropout": self.dropout,
            "normalization": self.normalization,
            "activation": self.activation,
            "convolution_mode": self.convolution_mode,
            "parameter_initialization": self.parameter_initialization,
            "mask_rule": self.mask_rule,
            "subject_adapter": {
                "enabled": self.subject_adapter.enabled,
                "type": self.subject_adapter.adapter_type,
                "num_subjects": self.subject_adapter.num_subjects,
                "gamma_initialization": (
                    self.subject_adapter.gamma_initialization
                ),
                "beta_initialization": (
                    self.subject_adapter.beta_initialization
                ),
            },
        }

    @property
    def canonical_sha256(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(encoded).hexdigest()


@dataclass(frozen=True)
class EEGSequenceOutput:
    sequence: torch.Tensor
    mask: torch.Tensor
    lengths: torch.Tensor


class EEGSequenceEncoder(nn.Module):
    """Padding-safe non-causal Conv1d encoder that preserves time."""

    def __init__(self, config: EEGSequenceEncoderConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.channel_projection = nn.Conv1d(
            config.input_channels,
            config.hidden_dim,
            kernel_size=1,
        )
        self.subject_adapter = (
            SubjectFiLM(
                num_subjects=config.subject_adapter.num_subjects,
                feature_dim=config.hidden_dim,
            )
            if config.subject_adapter.enabled
            else None
        )
        self.stem = MaskedTemporalDownsample(
            feature_dim=config.hidden_dim,
            kernel_size=config.stem_kernel_size,
            stride=config.stem_stride,
            dropout=config.dropout,
            activation=config.activation,
        )
        self.residual_blocks = nn.ModuleList(
            [
                MaskedDilatedResidualBlock(
                    hidden_dim=config.hidden_dim,
                    kernel_size=config.residual_kernel_size,
                    dilation=dilation,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for dilation in config.dilations
            ]
        )
        self.downsample = MaskedTemporalDownsample(
            feature_dim=config.hidden_dim,
            kernel_size=config.downsample_kernel_size,
            stride=config.downsample_stride,
            dropout=config.dropout,
            activation=config.activation,
        )
        self.output_projection = nn.Conv1d(
            config.hidden_dim,
            config.output_dim,
            kernel_size=1,
        )
        self.output_norm = TimewiseLayerNorm(config.output_dim)

    def _validate_inputs(
        self,
        eeg: torch.Tensor,
        eeg_mask: torch.Tensor,
        subject_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        if eeg.ndim != 3:
            raise ValueError(
                f"eeg must have shape [batch, channel, time], got "
                f"{tuple(eeg.shape)}"
            )
        if not eeg.is_floating_point():
            raise ValueError(f"eeg must be floating point, got {eeg.dtype}")
        batch_size, channels, time = eeg.shape
        if batch_size <= 0 or time <= 0:
            raise ValueError(f"eeg has an empty axis: {tuple(eeg.shape)}")
        if channels != self.config.input_channels:
            raise ValueError(
                f"EEG input channels {channels} do not match configured "
                f"{self.config.input_channels}"
            )
        if eeg_mask.shape != (batch_size, time):
            raise ValueError(
                f"eeg_mask must have shape {(batch_size, time)}, got "
                f"{tuple(eeg_mask.shape)}"
            )
        if eeg_mask.device != eeg.device:
            raise ValueError("eeg and eeg_mask must be on the same device")
        lengths = lengths_from_prefix_mask(eeg_mask)
        if int(lengths.min()) < self.config.minimum_input_length:
            raise ValueError(
                "EEG sequence shorter than configured minimum "
                f"{self.config.minimum_input_length}: {lengths.tolist()}"
            )
        if subject_indices is not None:
            if subject_indices.shape != (batch_size,):
                raise ValueError(
                    "subject_indices must have shape [batch], got "
                    f"{tuple(subject_indices.shape)}"
                )
            if subject_indices.dtype != torch.int64:
                raise ValueError("subject_indices must be torch.int64")
            if subject_indices.device != eeg.device:
                raise ValueError(
                    "subject_indices and eeg must be on the same device"
                )
        if self.subject_adapter is not None and subject_indices is None:
            raise ValueError(
                "subject_indices are required when the subject adapter is "
                "enabled"
            )
        return lengths

    def forward(
        self,
        *,
        eeg: torch.Tensor,
        eeg_mask: torch.Tensor,
        subject_indices: torch.Tensor | None,
    ) -> EEGSequenceOutput:
        lengths = self._validate_inputs(eeg, eeg_mask, subject_indices)
        sequence = mask_channel_first(eeg, eeg_mask)
        sequence = self.channel_projection(sequence)
        sequence = mask_channel_first(sequence, eeg_mask)
        if self.subject_adapter is not None:
            assert subject_indices is not None
            sequence = self.subject_adapter(sequence, subject_indices)
            sequence = mask_channel_first(sequence, eeg_mask)

        sequence, mask, lengths = self.stem(
            sequence,
            eeg_mask,
            lengths,
        )
        for block in self.residual_blocks:
            sequence = block(sequence, mask)
        sequence, mask, lengths = self.downsample(
            sequence,
            mask,
            lengths,
        )
        sequence = self.output_projection(sequence)
        sequence = mask_channel_first(sequence, mask)
        sequence = self.output_norm(sequence)
        sequence = mask_channel_first(sequence, mask)
        time_first = sequence.transpose(1, 2).contiguous()
        time_first = mask_time_first(time_first, mask)
        return EEGSequenceOutput(
            sequence=time_first,
            mask=mask,
            lengths=lengths,
        )

    def to_config(self) -> dict[str, Any]:
        return self.config.to_dict()

    @property
    def total_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


def build_eeg_sequence_encoder(
    config_path: str | Path,
    *,
    actual_input_channels: int | None = None,
    actual_num_subjects: int | None = None,
) -> EEGSequenceEncoder:
    config = EEGSequenceEncoderConfig.from_yaml(config_path)
    if (
        actual_input_channels is not None
        and config.input_channels != actual_input_channels
    ):
        raise ValueError(
            f"Config input_channels={config.input_channels} does not match "
            f"audited Dataset channels={actual_input_channels}"
        )
    if (
        actual_num_subjects is not None
        and config.subject_adapter.num_subjects != actual_num_subjects
    ):
        raise ValueError(
            "Config subject_adapter.num_subjects="
            f"{config.subject_adapter.num_subjects} does not match audited "
            f"Dataset subjects={actual_num_subjects}"
        )
    return EEGSequenceEncoder(config)
