from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import torch
from torch import nn


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_state_sha256(module: nn.Module) -> str:
    """Hash a module state independently of its current device."""

    digest = sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().to("cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(",".join(str(item) for item in tensor.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()
