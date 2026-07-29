from __future__ import annotations

from dataclasses import dataclass

import torch

from eeg_keyword_decoding.models import ContextTextProjector
from eeg_keyword_decoding.prototypes import (
    PrototypeBank,
    TrainOnlyPrototypeBuilder,
    module_state_sha256,
    prototype_bank_sha256,
)


@dataclass(frozen=True)
class PrototypeRefreshResult:
    bank: PrototypeBank
    bank_hash: str
    projector_hash: str


class PrototypeRefreshCoordinator:
    def __init__(self, builder: TrainOnlyPrototypeBuilder) -> None:
        self.builder = builder

    def refresh(
        self,
        projector: ContextTextProjector,
    ) -> PrototypeRefreshResult:
        before_training = projector.training
        projector.eval()
        try:
            with torch.inference_mode():
                bank = self.builder.build(
                    projector=projector,
                    role="train",
                ).detached()
        finally:
            projector.train(before_training)
        if projector.training != before_training:
            raise AssertionError("Prototype builder did not restore projector mode")
        projector_hash = module_state_sha256(projector)
        if bank.projector_state_hash != projector_hash:
            raise RuntimeError("Refreshed prototype bank has a stale projector hash")
        if bank.vectors.requires_grad:
            raise RuntimeError("Prototype refresh must return detached vectors")
        return PrototypeRefreshResult(
            bank=bank,
            bank_hash=prototype_bank_sha256(bank),
            projector_hash=projector_hash,
        )

    @staticmethod
    def assert_current(
        bank: PrototypeBank,
        projector: ContextTextProjector,
    ) -> None:
        current = module_state_sha256(projector)
        if bank.projector_state_hash != current:
            raise RuntimeError(
                "Prototype/projector hash mismatch: validation with a stale bank is forbidden"
            )
