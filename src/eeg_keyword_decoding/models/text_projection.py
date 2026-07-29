from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from torch import nn


TEXT_PROJECTION_CONFIG_SCHEMA = "context_text_projection_v1"
TextBackend = Literal["macbert", "bge_m3"]


@dataclass(frozen=True)
class TextProjectionOutput:
    sequence: torch.Tensor
    mask: torch.Tensor
    scalar_mix_weights: torch.Tensor | None


@dataclass(frozen=True)
class TextProjectionConfig:
    schema_version: str
    backend: TextBackend
    cache_metadata_path: str
    input_dim: int
    output_dim: int
    normalization: str
    projection: str
    scalar_mix_enabled: bool
    scalar_mix_layer_count: int
    scalar_mix_initialization: str
    expected_model_id: str
    expected_model_revision: str
    expected_context_vectors_sha256: str
    expected_storage_dtype: str
    expected_layer_indices: tuple[int, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TextProjectionConfig:
        scalar_mix = dict(value["scalar_mix"])
        expected = dict(value["expected_cache"])
        config = cls(
            schema_version=str(value["schema_version"]),
            backend=str(value["backend"]),  # type: ignore[arg-type]
            cache_metadata_path=str(value["cache_metadata_path"]),
            input_dim=int(value["input_dim"]),
            output_dim=int(value["output_dim"]),
            normalization=str(value["normalization"]),
            projection=str(value["projection"]),
            scalar_mix_enabled=bool(scalar_mix["enabled"]),
            scalar_mix_layer_count=int(scalar_mix["layer_count"]),
            scalar_mix_initialization=str(scalar_mix["initialization"]),
            expected_model_id=str(expected["model_id"]),
            expected_model_revision=str(expected["resolved_revision"]),
            expected_context_vectors_sha256=str(
                expected["context_vectors_sha256"]
            ),
            expected_storage_dtype=str(expected["storage_dtype"]),
            expected_layer_indices=tuple(
                int(item) for item in expected.get("layer_indices", [])
            ),
        )
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> TextProjectionConfig:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Text projection YAML root must be a mapping")
        return cls.from_dict(value)

    def validate(self) -> None:
        if self.schema_version != TEXT_PROJECTION_CONFIG_SCHEMA:
            raise ValueError(
                f"Unsupported text projection schema: {self.schema_version!r}"
            )
        if self.backend not in {"macbert", "bge_m3"}:
            raise ValueError(f"Unsupported text backend: {self.backend!r}")
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.output_dim != 256:
            raise ValueError("v1 freezes text projection output_dim at 256")
        if self.normalization != "layer_norm":
            raise ValueError("v1 requires layer_norm")
        if self.projection != "linear":
            raise ValueError("v1 requires a linear projection")
        if self.scalar_mix_initialization != "zeros":
            raise ValueError("v1 scalar mix logits must initialize to zero")
        if not self.expected_model_id or not self.expected_model_revision:
            raise ValueError("Expected cache model identity must be pinned")
        if len(self.expected_context_vectors_sha256) != 64:
            raise ValueError(
                "expected_context_vectors_sha256 must be a SHA256 digest"
            )
        if self.backend == "macbert":
            if not self.scalar_mix_enabled:
                raise ValueError("MacBERT requires scalar mix")
            if self.scalar_mix_layer_count <= 0:
                raise ValueError("MacBERT scalar mix layer_count must be positive")
            if (
                len(self.expected_layer_indices)
                != self.scalar_mix_layer_count
            ):
                raise ValueError(
                    "MacBERT expected layer order must match layer_count"
                )
        else:
            if self.scalar_mix_enabled or self.scalar_mix_layer_count != 0:
                raise ValueError("BGE-M3 must not enable scalar mix")
            if self.expected_layer_indices:
                raise ValueError("BGE-M3 must not declare encoder layers")

    def validate_cache_metadata(
        self,
        metadata: dict[str, Any],
    ) -> None:
        descriptor = dict(metadata.get("arrays", {})).get(
            "context_vectors"
        )
        if descriptor is None:
            raise ValueError("Cache metadata has no context_vectors descriptor")
        descriptor = dict(descriptor)
        if (
            str(descriptor["sha256"])
            != self.expected_context_vectors_sha256
        ):
            raise ValueError(
                "Context vector SHA256 disagrees with projection config"
            )
        if str(descriptor["dtype"]) != self.expected_storage_dtype:
            raise ValueError(
                f"Cache dtype {descriptor['dtype']} does not match configured "
                f"{self.expected_storage_dtype}"
            )
        model = dict(metadata["model"])
        extraction = dict(metadata["extraction"])
        if (
            extraction.get("model_eval") is not True
            or extraction.get("inference_mode") is not True
        ):
            raise ValueError(
                "Context cache must come from frozen eval/inference extraction"
            )
        if extraction.get("split_or_eeg_information_included") is not False:
            raise ValueError(
                "Context cache must not contain split or EEG information"
            )
        if str(model.get("model_id")) != self.expected_model_id:
            raise ValueError("Cache model_id disagrees with projection config")
        if (
            str(model.get("resolved_revision"))
            != self.expected_model_revision
        ):
            raise ValueError(
                "Cache model revision disagrees with projection config"
            )

        axis_order = tuple(extraction.get("output_axis_order", []))
        descriptor_shape = tuple(int(item) for item in descriptor["shape"])
        if self.backend == "macbert":
            expected_tail = (
                self.scalar_mix_layer_count,
                self.input_dim,
            )
            if descriptor_shape[1:] != expected_tail:
                raise ValueError(
                    "MacBERT cache shape disagrees with configured "
                    f"[layer, hidden] tail: {descriptor_shape[1:]} != "
                    f"{expected_tail}"
                )
            if axis_order != ("word", "encoder_layer", "hidden"):
                raise ValueError(
                    f"Unexpected MacBERT cache axis order: {axis_order}"
                )
            observed_layers = tuple(
                int(item)
                for item in extraction.get(
                    "hidden_state_layer_indices",
                    [],
                )
            )
            if observed_layers != self.expected_layer_indices:
                raise ValueError(
                    "MacBERT cache layer order disagrees with projection "
                    f"config: {observed_layers} != "
                    f"{self.expected_layer_indices}"
                )
        else:
            expected_tail = (self.input_dim,)
            if descriptor_shape[1:] != expected_tail:
                raise ValueError(
                    "BGE-M3 cache shape disagrees with configured hidden "
                    f"dimension: {descriptor_shape[1:]} != {expected_tail}"
                )
            if axis_order != ("word", "hidden"):
                raise ValueError(
                    f"Unexpected BGE-M3 cache axis order: {axis_order}"
                )
            colbert_dim = int(
                model.get(
                    "colbert_dimension",
                    model.get("hidden_size", -1),
                )
            )
            if colbert_dim != self.input_dim:
                raise ValueError(
                    "BGE-M3 ColBERT dimension disagrees with projection "
                    f"config: {colbert_dim} != {self.input_dim}"
                )
            representation = str(extraction.get("representation", ""))
            if not representation.lower().startswith("colbert"):
                raise ValueError(
                    "BGE-M3 cache must contain ColBERT token representations"
                )
            if extraction.get("dense_sentence_vectors_used") is not False:
                raise ValueError(
                    "BGE-M3 dense sentence vectors are forbidden"
                )

    @property
    def canonical_sha256(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "backend": self.backend,
            "cache_metadata_path": self.cache_metadata_path,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "normalization": self.normalization,
            "projection": self.projection,
            "scalar_mix": {
                "enabled": self.scalar_mix_enabled,
                "layer_count": self.scalar_mix_layer_count,
                "initialization": self.scalar_mix_initialization,
            },
            "expected_cache": {
                "model_id": self.expected_model_id,
                "resolved_revision": self.expected_model_revision,
                "context_vectors_sha256": (
                    self.expected_context_vectors_sha256
                ),
                "storage_dtype": self.expected_storage_dtype,
                "layer_indices": list(self.expected_layer_indices),
            },
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        import hashlib

        return hashlib.sha256(encoded).hexdigest()


def _validate_word_mask(
    word_mask: torch.Tensor,
    *,
    batch_size: int,
    word_count: int,
) -> None:
    if word_mask.shape != (batch_size, word_count):
        raise ValueError(
            f"word_mask must have shape {(batch_size, word_count)}, got "
            f"{tuple(word_mask.shape)}"
        )
    if word_mask.dtype != torch.bool:
        raise ValueError(f"word_mask must be torch.bool, got {word_mask.dtype}")
    lengths = word_mask.sum(dim=1, dtype=torch.int64)
    if bool((lengths == 0).any()):
        raise ValueError("Empty word sequences are forbidden")
    expected = (
        torch.arange(word_count, device=word_mask.device).unsqueeze(0)
        < lengths.unsqueeze(1)
    )
    if not torch.equal(word_mask, expected):
        raise ValueError(
            "word_mask must be a contiguous true prefix followed by padding"
        )


class ContextTextProjector(nn.Module):
    """Backend-specific frozen-feature projection into the 256-D OT space."""

    def __init__(self, config: TextProjectionConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        if config.backend == "macbert":
            self.layer_logits = nn.Parameter(
                torch.zeros(config.scalar_mix_layer_count)
            )
        else:
            self.register_parameter("layer_logits", None)
        self.normalization = nn.LayerNorm(config.input_dim)
        self.projection = nn.Linear(config.input_dim, config.output_dim)

    def forward(
        self,
        *,
        context_words: torch.Tensor,
        word_mask: torch.Tensor,
        backend: str | None = None,
    ) -> TextProjectionOutput:
        if backend is not None and backend != self.config.backend:
            raise ValueError(
                f"Batch backend {backend!r} does not match projector backend "
                f"{self.config.backend!r}"
            )
        if not context_words.is_floating_point():
            raise ValueError(
                f"context_words must be floating point, got "
                f"{context_words.dtype}"
            )
        if context_words.device != word_mask.device:
            raise ValueError(
                "context_words and word_mask must be on the same device"
            )
        if not bool(torch.isfinite(context_words).all()):
            raise ValueError("context_words contains NaN or Inf")

        if self.config.backend == "macbert":
            expected_rank = 4
            if context_words.ndim != expected_rank:
                raise ValueError(
                    "MacBERT context_words must have shape "
                    "[batch, word, layer, hidden]"
                )
            batch_size, word_count, layer_count, hidden_dim = (
                context_words.shape
            )
            if (
                layer_count != self.config.scalar_mix_layer_count
                or hidden_dim != self.config.input_dim
            ):
                raise ValueError(
                    "MacBERT input shape disagrees with projection config: "
                    f"{tuple(context_words.shape)}"
                )
            _validate_word_mask(
                word_mask,
                batch_size=batch_size,
                word_count=word_count,
            )
            assert self.layer_logits is not None
            weights = torch.softmax(self.layer_logits, dim=0)
            mixed = torch.einsum(
                "l,bnld->bnd",
                weights,
                context_words,
            )
        else:
            expected_rank = 3
            if context_words.ndim != expected_rank:
                raise ValueError(
                    "BGE-M3 context_words must have shape "
                    "[batch, word, hidden]"
                )
            batch_size, word_count, hidden_dim = context_words.shape
            if hidden_dim != self.config.input_dim:
                raise ValueError(
                    "BGE-M3 input shape disagrees with projection config: "
                    f"{tuple(context_words.shape)}"
                )
            _validate_word_mask(
                word_mask,
                batch_size=batch_size,
                word_count=word_count,
            )
            weights = None
            mixed = context_words

        mixed = mixed.masked_fill(~word_mask.unsqueeze(-1), 0.0)
        sequence = self.normalization(mixed)
        sequence = sequence.masked_fill(~word_mask.unsqueeze(-1), 0.0)
        sequence = self.projection(sequence)
        sequence = sequence.masked_fill(~word_mask.unsqueeze(-1), 0.0)
        if not bool(torch.isfinite(sequence).all()):
            raise FloatingPointError("Text projection produced NaN or Inf")
        return TextProjectionOutput(
            sequence=sequence,
            mask=word_mask,
            scalar_mix_weights=weights,
        )

    @property
    def total_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def _find_project_root(path: Path) -> Path:
    for candidate in (path.parent, *path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd()


def build_context_text_projector(
    config_path: str | Path,
    *,
    cache_metadata_path: str | Path | None = None,
) -> ContextTextProjector:
    config_file = Path(config_path).resolve()
    config = TextProjectionConfig.from_yaml(config_file)
    metadata_path = (
        Path(cache_metadata_path)
        if cache_metadata_path is not None
        else Path(config.cache_metadata_path)
    )
    if not metadata_path.is_absolute():
        metadata_path = _find_project_root(config_file) / metadata_path
    metadata = json.loads(
        metadata_path.resolve().read_text(encoding="utf-8")
    )
    if not isinstance(metadata, dict):
        raise ValueError("Cache metadata root must be a mapping")
    config.validate_cache_metadata(metadata)
    return ContextTextProjector(config)
