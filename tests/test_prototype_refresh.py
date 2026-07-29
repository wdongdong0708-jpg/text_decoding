from __future__ import annotations

import torch
from torch.nn import functional as F

from eeg_keyword_decoding.models import ContextTextProjector, TextProjectionConfig
from eeg_keyword_decoding.prototypes import (
    PrototypeBank,
    module_state_sha256,
    prototype_bank_sha256,
)
from eeg_keyword_decoding.training import PrototypeRefreshCoordinator


def _projector() -> ContextTextProjector:
    return ContextTextProjector(
        TextProjectionConfig(
            schema_version="context_text_projection_v1",
            backend="bge_m3",
            cache_metadata_path="unused",
            input_dim=8,
            output_dim=256,
            normalization="layer_norm",
            projection="linear",
            scalar_mix_enabled=False,
            scalar_mix_layer_count=0,
            scalar_mix_initialization="zeros",
            expected_model_id="fixture",
            expected_model_revision="revision",
            expected_context_vectors_sha256="a" * 64,
            expected_storage_dtype="float16",
            expected_layer_indices=(),
        )
    )


class _Builder:
    def build(self, *, projector, role):
        assert role == "train"
        assert not projector.training
        # Project fixed train text only; no EEG inputs exist in this fixture.
        source = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)
        output = projector(
            context_words=source, word_mask=torch.ones(1, 1, dtype=torch.bool)
        ).sequence[0, 0]
        vectors = torch.zeros(247, 256)
        vectors[0] = F.normalize(output.float(), dim=0)
        digest = "b" * 64
        return PrototypeBank(
            vectors=vectors,
            available_mask=torch.tensor([True] + [False] * 246),
            keyword_ids=tuple(f"kw-{index}" for index in range(247)),
            train_sentence_df=torch.ones(247, dtype=torch.int64),
            train_group_df=torch.ones(247, dtype=torch.int64),
            outer_fold=0,
            text_backend="bge_m3",
            projector_state_hash=module_state_sha256(projector),
            source_cache_hash=digest,
            source_cache_metadata_hash=digest,
            fold_hash=digest,
            eligibility_hash=digest,
            lexical_mapping_hash=digest,
            metadata={"contributors": {"text_embedding_indices": [0]}},
        )


def test_refresh_is_deterministic_detached_and_restores_mode() -> None:
    projector = _projector()
    projector.train()
    coordinator = PrototypeRefreshCoordinator(_Builder())  # type: ignore[arg-type]
    first = coordinator.refresh(projector)
    second = coordinator.refresh(projector)
    assert projector.training
    assert first.bank_hash == second.bank_hash
    assert first.bank.projector_state_hash == module_state_sha256(projector)
    assert not first.bank.vectors.requires_grad
    assert first.bank.vectors.dtype == torch.float32


def test_old_bank_is_rejected_after_projector_update() -> None:
    projector = _projector()
    coordinator = PrototypeRefreshCoordinator(_Builder())  # type: ignore[arg-type]
    old = coordinator.refresh(projector)
    with torch.no_grad():
        projector.projection.weight.add_(0.01)
    try:
        coordinator.assert_current(old.bank, projector)
    except RuntimeError as error:
        assert "stale bank" in str(error)
    else:
        raise AssertionError("A stale prototype bank was accepted")
    new = coordinator.refresh(projector)
    assert new.projector_hash != old.projector_hash
    assert prototype_bank_sha256(new.bank) != prototype_bank_sha256(old.bank)
