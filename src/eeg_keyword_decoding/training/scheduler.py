from __future__ import annotations

import math
from dataclasses import dataclass

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


@dataclass(frozen=True)
class SchedulerConfig:
    name: str = "linear_warmup_cosine"
    total_optimizer_steps: int = 1
    warmup_steps: int = 0


def build_scheduler(
    optimizer: Optimizer,
    config: SchedulerConfig,
) -> LambdaLR:
    if config.total_optimizer_steps <= 0:
        raise ValueError("total_optimizer_steps must be positive")
    if not 0 <= config.warmup_steps < config.total_optimizer_steps:
        raise ValueError("warmup_steps must be in [0,total_optimizer_steps)")
    if config.name not in {"linear_warmup_cosine", "constant_with_warmup"}:
        raise ValueError(f"Unsupported scheduler: {config.name}")

    def factor(step: int) -> float:
        if config.warmup_steps and step < config.warmup_steps:
            return float(step + 1) / float(config.warmup_steps)
        if config.name == "constant_with_warmup":
            return 1.0
        decay_steps = config.total_optimizer_steps - config.warmup_steps
        progress = min(
            1.0,
            max(0.0, (step - config.warmup_steps) / max(1, decay_steps)),
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=factor)
