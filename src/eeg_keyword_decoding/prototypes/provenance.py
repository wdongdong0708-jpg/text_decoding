from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import torch
from torch import nn

from .schema import PrototypeBank


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_state_sha256(module: nn.Module) -> str:
    """Hash a module state independently of its current device."""

    return state_dict_sha256(module.state_dict())


def state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    """Hash a tensor state mapping independently of its current device."""

    digest = sha256()
    for name, value in sorted(state_dict.items()):
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


def prototype_bank_sha256(bank: PrototypeBank) -> str:
    """Hash the immutable scorer content and identity of a prototype bank."""

    bank.validate()
    digest = sha256()
    for name, value in (
        ("vectors", bank.vectors),
        ("available_mask", bank.available_mask),
        ("train_sentence_df", bank.train_sentence_df),
        ("train_group_df", bank.train_group_df),
    ):
        tensor = value.detach().to("cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    for value in (
        str(bank.outer_fold),
        bank.text_backend,
        bank.projector_state_hash,
        *bank.keyword_ids,
    ):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()
