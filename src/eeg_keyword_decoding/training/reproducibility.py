from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class ReproducibilityConfig:
    seed: int = 42
    deterministic_algorithms: bool = True
    cudnn_benchmark: bool = False


def configure_reproducibility(config: ReproducibilityConfig) -> None:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(
        config.deterministic_algorithms,
        warn_only=False,
    )
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = config.cudnn_benchmark
        torch.backends.cudnn.deterministic = config.deterministic_algorithms


def make_dataloader_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def seed_dataloader_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def capture_rng_state(
    *,
    dataloader_generator: torch.Generator | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }
    if dataloader_generator is not None:
        state["dataloader_generator"] = dataloader_generator.get_state()
    return state


def restore_rng_state(
    state: dict[str, Any],
    *,
    dataloader_generator: torch.Generator | None = None,
) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(
            [value.cpu() for value in state["torch_cuda"]]
        )
    if dataloader_generator is not None:
        if "dataloader_generator" not in state:
            raise ValueError("Checkpoint has no DataLoader generator state")
        dataloader_generator.set_state(state["dataloader_generator"].cpu())
