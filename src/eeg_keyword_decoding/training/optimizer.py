from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from eeg_keyword_decoding.models import ContextTextProjector


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "adamw"
    eeg_lr: float = 3.0e-4
    text_projection_lr: float = 1.0e-4
    scalar_mix_lr: float = 1.0e-4
    other_lr: float = 1.0e-4
    weight_decay: float = 1.0e-2
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1.0e-8


def _decay_allowed(name: str, parameter: nn.Parameter) -> bool:
    leaf = name.rsplit(".", 1)[-1].lower()
    return parameter.ndim > 1 and leaf != "bias"


def _category_parameters(
    eeg_encoder: nn.Module,
    text_projector: ContextTextProjector,
    other_modules: Iterable[nn.Module],
) -> tuple[tuple[str, list[tuple[str, nn.Parameter]], float], ...]:
    eeg = [(f"eeg_encoder.{name}", value) for name, value in eeg_encoder.named_parameters()]
    scalar: list[tuple[str, nn.Parameter]] = []
    text: list[tuple[str, nn.Parameter]] = []
    for name, value in text_projector.named_parameters():
        target = scalar if name == "layer_logits" else text
        target.append((f"text_projector.{name}", value))
    other: list[tuple[str, nn.Parameter]] = []
    for module_index, module in enumerate(other_modules):
        other.extend(
            (f"other_{module_index}.{name}", value)
            for name, value in module.named_parameters()
        )
    return (
        ("eeg", eeg, 0.0),
        ("text_projection", text, 0.0),
        ("scalar_mix", scalar, 0.0),
        ("other", other, 0.0),
    )


def build_adamw_optimizer(
    *,
    eeg_encoder: nn.Module,
    text_projector: ContextTextProjector,
    config: OptimizerConfig,
    other_modules: Iterable[nn.Module] = (),
) -> torch.optim.AdamW:
    if config.name.lower() != "adamw":
        raise ValueError("Stage 6A supports AdamW only")
    if min(
        config.eeg_lr,
        config.text_projection_lr,
        config.scalar_mix_lr,
        config.other_lr,
    ) <= 0:
        raise ValueError("All optimizer learning rates must be positive")
    category_lrs = {
        "eeg": config.eeg_lr,
        "text_projection": config.text_projection_lr,
        "scalar_mix": config.scalar_mix_lr,
        "other": config.other_lr,
    }
    categories = _category_parameters(eeg_encoder, text_projector, other_modules)
    groups: list[dict[str, object]] = []
    seen: set[int] = set()
    expected: set[int] = set()
    for category, named_parameters, _ in categories:
        decay: list[nn.Parameter] = []
        no_decay: list[nn.Parameter] = []
        for name, parameter in named_parameters:
            if not parameter.requires_grad:
                continue
            identity = id(parameter)
            expected.add(identity)
            if identity in seen:
                raise ValueError(f"Trainable parameter appears more than once: {name}")
            seen.add(identity)
            (decay if _decay_allowed(name, parameter) else no_decay).append(parameter)
        for suffix, parameters, weight_decay in (
            ("decay", decay, config.weight_decay),
            ("no_decay", no_decay, 0.0),
        ):
            if parameters:
                groups.append(
                    {
                        "params": parameters,
                        "lr": category_lrs[category],
                        "weight_decay": weight_decay,
                        "group_name": f"{category}/{suffix}",
                    }
                )
    if seen != expected:
        raise AssertionError("Optimizer coverage accounting failed")
    optimizer = torch.optim.AdamW(
        groups,
        betas=config.betas,
        eps=config.eps,
    )
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if optimizer_ids != expected:
        raise AssertionError("Every trainable parameter must appear exactly once")
    return optimizer
