from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from eeg_keyword_decoding.data import ContextEEGBatch
from eeg_keyword_decoding.losses import ContextOTThreeLoss
from eeg_keyword_decoding.ot import ContextOTAligner
from eeg_keyword_decoding.prototypes import PrototypeBank

from .logging import JsonlLogger


@dataclass(frozen=True)
class TrainEpochConfig:
    precision: str = "amp"
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    max_batches: int | None = None
    log_every_optimizer_steps: int = 1


@dataclass(frozen=True)
class TrainEpochResult:
    micro_batches: int
    optimizer_steps: int
    global_optimizer_step: int
    mean_total_loss: float
    mean_ot_loss: float
    mean_token_loss: float
    mean_prototype_loss: float
    final_gradient_norm: float
    elapsed_seconds: float
    peak_gpu_memory_bytes: int


def _assert_finite_gradients(modules: Iterable[nn.Module], batch_ids: tuple[str, ...]) -> None:
    for module in modules:
        for name, parameter in module.named_parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError(
                    f"Non-finite gradient in {name}; batch EEG views={batch_ids}"
                )


def train_one_epoch(
    *,
    epoch: int,
    global_optimizer_step: int,
    eeg_encoder: nn.Module,
    aligner: ContextOTAligner,
    loss_module: ContextOTThreeLoss,
    train_loader: Iterable[ContextEEGBatch],
    prototype_bank: PrototypeBank,
    optimizer: torch.optim.Optimizer,
    scheduler: object | None,
    scaler: torch.amp.GradScaler,
    device: torch.device | str,
    config: TrainEpochConfig,
    logger: JsonlLogger | None = None,
) -> TrainEpochResult:
    if config.precision not in {"amp", "fp32"}:
        raise ValueError("precision must be 'amp' or 'fp32'")
    if config.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if config.gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive")
    target = torch.device(device)
    eeg_encoder.train()
    aligner.train()
    bank = prototype_bank.to(target)
    optimizer.zero_grad(set_to_none=True)
    if target.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target)
    loader_length = len(train_loader)  # type: ignore[arg-type]
    batch_count = (
        loader_length
        if config.max_batches is None
        else min(loader_length, config.max_batches)
    )
    if batch_count <= 0:
        raise ValueError("Training loader has no selected batches")

    totals: list[float] = []
    ots: list[float] = []
    tokens: list[float] = []
    prototypes: list[float] = []
    optimizer_steps = 0
    final_gradient_norm = 0.0
    start_time = time.perf_counter()
    group_start = 0
    group_size = min(config.gradient_accumulation_steps, batch_count)
    for micro_index, batch in enumerate(train_loader):
        if micro_index >= batch_count:
            break
        if batch.context_words is None:
            raise RuntimeError("Training batch has no context targets")
        if set(batch.roles) != {"train"}:
            raise RuntimeError("Training loader yielded a non-train role")
        if micro_index == group_start:
            group_size = min(
                config.gradient_accumulation_steps,
                batch_count - group_start,
            )
        moved = batch.to(target, non_blocking=True)
        amp_enabled = config.precision == "amp" and target.type == "cuda"
        with torch.autocast(device_type=target.type, enabled=amp_enabled):
            encoded = eeg_encoder(
                eeg=moved.eeg,
                eeg_mask=moved.eeg_mask,
                subject_indices=moved.subject_indices,
            )
            alignment = aligner(
                eeg_sequence=encoded.sequence,
                eeg_mask=encoded.mask,
                context_words=moved.context_words,
                word_mask=moved.word_mask,
                context_backend=str(moved.context_backend),
            )
            loss = loss_module(
                alignment=alignment,
                word_mask=moved.word_mask,
                context_token_group_indices=moved.context_token_group_indices,
                surface_type_indices=moved.surface_type_indices,
                sentence_group_indices=moved.sentence_group_indices,
                word_keyword_indices=moved.word_keyword_indices,
                prototype_bank=bank,
            )
        if not bool(torch.isfinite(loss.total)):
            raise FloatingPointError(
                f"Non-finite loss; batch EEG views={batch.eeg_view_ids}"
            )
        scaler.scale(loss.total / group_size).backward()
        totals.append(float(loss.total.detach().cpu()))
        ots.append(float(loss.ot_context.detach().cpu()))
        tokens.append(float(loss.context_token.detach().cpu()))
        prototypes.append(float(loss.prototype.detach().cpu()))

        at_group_end = (
            (micro_index - group_start + 1) == group_size
            or micro_index + 1 == batch_count
        )
        if at_group_end:
            # Required ordering: AMP unscale, finite-gradient audit, then clipping.
            scaler.unscale_(optimizer)
            _assert_finite_gradients(
                (eeg_encoder, aligner),
                batch.eeg_view_ids,
            )
            norm = torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for group in optimizer.param_groups
                    for parameter in group["params"]
                ],
                config.gradient_clip_norm,
                error_if_nonfinite=True,
            )
            final_gradient_norm = float(norm.detach().cpu())
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()  # type: ignore[attr-defined]
            optimizer_steps += 1
            global_optimizer_step += 1
            if (
                logger is not None
                and global_optimizer_step % config.log_every_optimizer_steps == 0
            ):
                logger.write(
                    {
                        "event": "train_step",
                        "epoch": epoch,
                        "micro_step": micro_index + 1,
                        "optimizer_step": global_optimizer_step,
                        "learning_rates": [
                            float(group["lr"]) for group in optimizer.param_groups
                        ],
                        "loss_total": totals[-1],
                        "loss_ot": ots[-1],
                        "loss_token": tokens[-1],
                        "loss_prototype": prototypes[-1],
                        "gradient_norm_before_clip": final_gradient_norm,
                        "amp_scale": float(scaler.get_scale()),
                        "eeg_length_min": int(batch.eeg_lengths.min()),
                        "eeg_length_max": int(batch.eeg_lengths.max()),
                        "word_count_min": int(batch.word_lengths.min()),
                        "word_count_max": int(batch.word_lengths.max()),
                        "prototype_available_count": bank.available_count,
                        "sinkhorn_row_error_max": loss.diagnostics[
                            "plan_row_error_max"
                        ],
                        "sinkhorn_column_error_max": loss.diagnostics[
                            "plan_column_error_max"
                        ],
                        "token_query_count": loss.diagnostics[
                            "token_valid_query_count"
                        ],
                        "scalar_mix_weights": loss.diagnostics[
                            "scalar_mix_weights"
                        ],
                    }
                )
            group_start = micro_index + 1
    elapsed = time.perf_counter() - start_time
    peak = (
        int(torch.cuda.max_memory_allocated(target))
        if target.type == "cuda"
        else 0
    )
    return TrainEpochResult(
        micro_batches=batch_count,
        optimizer_steps=optimizer_steps,
        global_optimizer_step=global_optimizer_step,
        mean_total_loss=sum(totals) / len(totals),
        mean_ot_loss=sum(ots) / len(ots),
        mean_token_loss=sum(tokens) / len(tokens),
        mean_prototype_loss=sum(prototypes) / len(prototypes),
        final_gradient_norm=final_gradient_norm,
        elapsed_seconds=elapsed,
        peak_gpu_memory_bytes=peak,
    )
