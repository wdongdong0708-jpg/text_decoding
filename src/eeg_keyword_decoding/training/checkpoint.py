from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from eeg_keyword_decoding.models import ContextTextProjector
from eeg_keyword_decoding.prototypes import (
    PrototypeBank,
    module_state_sha256,
    prototype_bank_sha256,
    state_dict_sha256,
)

from .early_stopping import EarlyStopping
from .reproducibility import capture_rng_state, restore_rng_state


CHECKPOINT_SCHEMA_VERSION = "fold_training_checkpoint_v1"


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _bank_to_payload(bank: PrototypeBank) -> dict[str, Any]:
    bank.validate()
    return {
        "vectors": bank.vectors.detach().cpu(),
        "available_mask": bank.available_mask.detach().cpu(),
        "keyword_ids": bank.keyword_ids,
        "train_sentence_df": bank.train_sentence_df.detach().cpu(),
        "train_group_df": bank.train_group_df.detach().cpu(),
        "outer_fold": bank.outer_fold,
        "text_backend": bank.text_backend,
        "projector_state_hash": bank.projector_state_hash,
        "source_cache_hash": bank.source_cache_hash,
        "source_cache_metadata_hash": bank.source_cache_metadata_hash,
        "fold_hash": bank.fold_hash,
        "eligibility_hash": bank.eligibility_hash,
        "lexical_mapping_hash": bank.lexical_mapping_hash,
        "metadata": bank.metadata,
    }


def _bank_from_payload(payload: Mapping[str, Any]) -> PrototypeBank:
    bank = PrototypeBank(
        vectors=payload["vectors"],
        available_mask=payload["available_mask"],
        keyword_ids=tuple(payload["keyword_ids"]),
        train_sentence_df=payload["train_sentence_df"],
        train_group_df=payload["train_group_df"],
        outer_fold=int(payload["outer_fold"]),
        text_backend=str(payload["text_backend"]),
        projector_state_hash=str(payload["projector_state_hash"]),
        source_cache_hash=str(payload["source_cache_hash"]),
        source_cache_metadata_hash=str(payload["source_cache_metadata_hash"]),
        fold_hash=str(payload["fold_hash"]),
        eligibility_hash=str(payload["eligibility_hash"]),
        lexical_mapping_hash=str(payload["lexical_mapping_hash"]),
        metadata=dict(payload["metadata"]),
    )
    bank.validate()
    return bank


def _environment_versions() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_names": (
            [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
            if torch.cuda.is_available()
            else []
        ),
    }


def _git_state(project_root: str | Path | None) -> dict[str, object]:
    if project_root is None:
        return {"commit": None, "dirty": None}
    root = Path(project_root)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


@dataclass(frozen=True)
class CheckpointExpectations:
    outer_fold: int
    text_backend: str
    input_channels: int
    subject_index_mapping: Mapping[str, int]
    keyword_index_mapping: Mapping[str, int]
    asset_hashes: Mapping[str, str]
    config_sha256: str


@dataclass(frozen=True)
class LoadedTrainingState:
    epoch: int
    global_optimizer_step: int
    prototype_bank: PrototypeBank
    current_validation_metrics: dict[str, float]
    best_validation_metrics: dict[str, float]
    dataloader_epoch_start_state: torch.Tensor | None


def save_training_checkpoint(
    path: str | Path,
    *,
    outer_fold: int,
    epoch: int,
    global_optimizer_step: int,
    eeg_encoder: nn.Module,
    text_projector: ContextTextProjector,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    prototype_bank: PrototypeBank,
    resolved_config: Mapping[str, Any],
    asset_hashes: Mapping[str, str],
    subject_index_mapping: Mapping[str, int],
    keyword_index_mapping: Mapping[str, int],
    current_validation_metrics: Mapping[str, float],
    best_validation_metrics: Mapping[str, float],
    early_stopping: EarlyStopping,
    dataloader_generator: torch.Generator | None = None,
    dataloader_epoch_start_state: torch.Tensor | None = None,
    project_root: str | Path | None = None,
) -> None:
    prototype_bank.validate()
    projector_hash = module_state_sha256(text_projector)
    if prototype_bank.projector_state_hash != projector_hash:
        raise RuntimeError("Cannot checkpoint mismatched prototype/projector states")
    config_copy = json.loads(json.dumps(resolved_config, default=str))
    scalar_mix_state = (
        None
        if text_projector.layer_logits is None
        else text_projector.layer_logits.detach().cpu()
    )
    payload: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "outer_fold": int(outer_fold),
        "epoch": int(epoch),
        "global_optimizer_step": int(global_optimizer_step),
        "text_backend": text_projector.config.backend,
        "input_channels": int(getattr(eeg_encoder.config, "input_channels")),
        "eeg_encoder_state": eeg_encoder.state_dict(),
        "text_projector_state": text_projector.state_dict(),
        "scalar_mix_state": scalar_mix_state,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "grad_scaler_state": scaler.state_dict() if scaler is not None else None,
        "prototype_bank": _bank_to_payload(prototype_bank),
        "prototype_bank_hash": prototype_bank_sha256(prototype_bank),
        "projector_state_hash": projector_hash,
        "resolved_config": config_copy,
        "config_sha256": canonical_sha256(config_copy),
        "asset_hashes": dict(asset_hashes),
        "context_cache": {
            "vectors_sha256": prototype_bank.source_cache_hash,
            "metadata_sha256": prototype_bank.source_cache_metadata_hash,
            "backend": prototype_bank.text_backend,
            "model_id": text_projector.config.expected_model_id,
            "model_revision": text_projector.config.expected_model_revision,
        },
        "loss_config": config_copy.get("loss"),
        "sinkhorn_config": config_copy.get("sinkhorn"),
        "scorer_config": config_copy.get("validation_scorer"),
        "subject_index_mapping": dict(subject_index_mapping),
        "keyword_index_mapping": dict(keyword_index_mapping),
        "current_validation_metrics": dict(current_validation_metrics),
        "best_validation_metrics": dict(best_validation_metrics),
        "early_stopping_state": early_stopping.state_dict(),
        "rng_state": capture_rng_state(dataloader_generator=dataloader_generator),
        "dataloader_epoch_start_state": dataloader_epoch_start_state,
        "environment_versions": _environment_versions(),
        "git": _git_state(project_root),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)


def _validate_payload(
    payload: Mapping[str, Any],
    expected: CheckpointExpectations,
) -> None:
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Unsupported checkpoint schema")
    checks = (
        ("outer_fold", int(payload["outer_fold"]), expected.outer_fold),
        ("text_backend", str(payload["text_backend"]), expected.text_backend),
        ("input_channels", int(payload["input_channels"]), expected.input_channels),
        (
            "subject_index_mapping",
            dict(payload["subject_index_mapping"]),
            dict(expected.subject_index_mapping),
        ),
        (
            "keyword_index_mapping",
            dict(payload["keyword_index_mapping"]),
            dict(expected.keyword_index_mapping),
        ),
        ("asset_hashes", dict(payload["asset_hashes"]), dict(expected.asset_hashes)),
        ("config_sha256", str(payload["config_sha256"]), expected.config_sha256),
    )
    for name, observed, wanted in checks:
        if observed != wanted:
            raise ValueError(f"Checkpoint {name} mismatch")
    if canonical_sha256(payload["resolved_config"]) != payload["config_sha256"]:
        raise ValueError("Checkpoint resolved config hash is corrupt")


def load_training_checkpoint(
    path: str | Path,
    *,
    expected: CheckpointExpectations,
    eeg_encoder: nn.Module,
    text_projector: ContextTextProjector,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    early_stopping: EarlyStopping,
    dataloader_generator: torch.Generator | None = None,
    map_location: str | torch.device = "cpu",
) -> LoadedTrainingState:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint root must be a mapping")
    _validate_payload(payload, expected)
    bank = _bank_from_payload(payload["prototype_bank"])
    if prototype_bank_sha256(bank) != payload["prototype_bank_hash"]:
        raise ValueError("Checkpoint prototype bank hash is corrupt")

    eeg_encoder.load_state_dict(payload["eeg_encoder_state"], strict=True)
    text_projector.load_state_dict(payload["text_projector_state"], strict=True)
    observed_projector_hash = module_state_sha256(text_projector)
    if observed_projector_hash != payload["projector_state_hash"]:
        raise ValueError("Checkpoint projector state hash is corrupt")
    if bank.projector_state_hash != observed_projector_hash:
        raise ValueError("Checkpoint prototype/projector hash mismatch")
    observed_scalar = (
        None
        if text_projector.layer_logits is None
        else text_projector.layer_logits.detach().cpu()
    )
    stored_scalar = payload["scalar_mix_state"]
    if (stored_scalar is None) != (observed_scalar is None) or (
        stored_scalar is not None
        and not torch.equal(stored_scalar, observed_scalar)
    ):
        raise ValueError("Checkpoint scalar-mix state is inconsistent")

    optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None:
        if payload["scheduler_state"] is None:
            raise ValueError("Checkpoint has no scheduler state")
        scheduler.load_state_dict(payload["scheduler_state"])
    if scaler is not None:
        if payload["grad_scaler_state"] is None:
            raise ValueError("Checkpoint has no GradScaler state")
        scaler.load_state_dict(payload["grad_scaler_state"])
    early_stopping.load_state_dict(payload["early_stopping_state"])
    restore_rng_state(
        payload["rng_state"],
        dataloader_generator=dataloader_generator,
    )
    return LoadedTrainingState(
        epoch=int(payload["epoch"]),
        global_optimizer_step=int(payload["global_optimizer_step"]),
        prototype_bank=bank,
        current_validation_metrics={
            key: float(value)
            for key, value in payload["current_validation_metrics"].items()
        },
        best_validation_metrics={
            key: float(value)
            for key, value in payload["best_validation_metrics"].items()
        },
        dataloader_epoch_start_state=payload.get("dataloader_epoch_start_state"),
    )


def inspect_checkpoint(path: str | Path) -> dict[str, Any]:
    """Read checkpoint metadata without constructing datasets or loading EEG."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    bank = _bank_from_payload(payload["prototype_bank"])
    if prototype_bank_sha256(bank) != payload["prototype_bank_hash"]:
        raise ValueError("Checkpoint prototype bank hash is corrupt")
    if canonical_sha256(payload["resolved_config"]) != payload["config_sha256"]:
        raise ValueError("Checkpoint resolved config hash is corrupt")
    projector_hash = state_dict_sha256(payload["text_projector_state"])
    if (
        projector_hash != payload["projector_state_hash"]
        or bank.projector_state_hash != projector_hash
    ):
        raise ValueError("Checkpoint prototype/projector hash mismatch")
    return {
        "schema_version": payload["schema_version"],
        "outer_fold": payload["outer_fold"],
        "epoch": payload["epoch"],
        "global_optimizer_step": payload["global_optimizer_step"],
        "text_backend": payload["text_backend"],
        "input_channels": payload["input_channels"],
        "prototype_available_count": bank.available_count,
        "prototype_bank_hash": payload["prototype_bank_hash"],
        "projector_state_hash": payload["projector_state_hash"],
        "config_sha256": payload["config_sha256"],
        "asset_hashes": payload["asset_hashes"],
        "current_validation_metrics": payload["current_validation_metrics"],
        "best_validation_metrics": payload["best_validation_metrics"],
        "early_stopping_state": payload["early_stopping_state"],
        "environment_versions": payload["environment_versions"],
        "git": payload["git"],
        "created_at_utc": payload["created_at_utc"],
        "integrity_validated": True,
    }
