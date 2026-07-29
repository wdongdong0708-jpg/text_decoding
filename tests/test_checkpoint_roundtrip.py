from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from eeg_keyword_decoding.models import (
    build_context_text_projector,
    build_eeg_sequence_encoder,
)
from eeg_keyword_decoding.prototypes import PrototypeBank, module_state_sha256
from eeg_keyword_decoding.training import (
    CheckpointExpectations,
    EarlyStopping,
    OptimizerConfig,
    SchedulerConfig,
    build_adamw_optimizer,
    build_scheduler,
    canonical_sha256,
    inspect_checkpoint,
    load_training_checkpoint,
    save_training_checkpoint,
)


def _objects():
    root = Path(__file__).resolve().parents[1]
    eeg = build_eeg_sequence_encoder(root / "configs/models/eeg_sequence_conv_v1.yaml")
    text = build_context_text_projector(root / "configs/text/bge_m3_projection_v1.yaml")
    optimizer = build_adamw_optimizer(
        eeg_encoder=eeg, text_projector=text, config=OptimizerConfig()
    )
    scheduler = build_scheduler(
        optimizer, SchedulerConfig(total_optimizer_steps=4, warmup_steps=1)
    )
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    return eeg, text, optimizer, scheduler, scaler


def _bank(text) -> PrototypeBank:
    generator = torch.Generator().manual_seed(7)
    vectors = F.normalize(torch.randn(247, 256, generator=generator), dim=-1)
    digest = "c" * 64
    return PrototypeBank(
        vectors=vectors,
        available_mask=torch.ones(247, dtype=torch.bool),
        keyword_ids=tuple(f"kw-{index}" for index in range(247)),
        train_sentence_df=torch.ones(247, dtype=torch.int64),
        train_group_df=torch.ones(247, dtype=torch.int64),
        outer_fold=0,
        text_backend="bge_m3",
        projector_state_hash=module_state_sha256(text),
        source_cache_hash=digest,
        source_cache_metadata_hash=digest,
        fold_hash=digest,
        eligibility_hash=digest,
        lexical_mapping_hash=digest,
        metadata={},
    )


def test_checkpoint_restores_full_state_rng_and_is_inspectable_without_eeg(tmp_path) -> None:
    eeg, text, optimizer, scheduler, scaler = _objects()
    config = {"trainer": {"precision": "fp32"}, "backend": "bge_m3"}
    assets = {"fold": "d" * 64, "manifest": "e" * 64}
    subjects = {"sub-01": 0}
    keywords = {f"kw-{index}": index for index in range(247)}
    early = EarlyStopping(patience=2)
    early.update(
        {"validation/core/context_group/macro_auprc": 0.25}, epoch=0
    )
    loader_generator = torch.Generator().manual_seed(44)
    path = tmp_path / "last.pt"
    save_training_checkpoint(
        path,
        outer_fold=0,
        epoch=0,
        global_optimizer_step=3,
        eeg_encoder=eeg,
        text_projector=text,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        prototype_bank=_bank(text),
        resolved_config=config,
        asset_hashes=assets,
        subject_index_mapping=subjects,
        keyword_index_mapping=keywords,
        current_validation_metrics={
            "validation/core/context_group/macro_auprc": 0.25
        },
        best_validation_metrics={
            "validation/core/context_group/macro_auprc": 0.25
        },
        early_stopping=early,
        dataloader_generator=loader_generator,
    )
    expected_python = random.random()
    expected_numpy = np.random.random()
    expected_torch = torch.rand(1)
    expected_loader = torch.rand(1, generator=loader_generator)
    with torch.no_grad():
        next(eeg.parameters()).add_(2.0)
        next(text.parameters()).add_(2.0)
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    loader_generator.manual_seed(999)

    loaded = load_training_checkpoint(
        path,
        expected=CheckpointExpectations(
            outer_fold=0,
            text_backend="bge_m3",
            input_channels=128,
            subject_index_mapping=subjects,
            keyword_index_mapping=keywords,
            asset_hashes=assets,
            config_sha256=canonical_sha256(config),
        ),
        eeg_encoder=eeg,
        text_projector=text,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        early_stopping=early,
        dataloader_generator=loader_generator,
    )
    assert loaded.global_optimizer_step == 3
    assert random.random() == expected_python
    assert np.random.random() == expected_numpy
    assert torch.equal(torch.rand(1), expected_torch)
    assert torch.equal(torch.rand(1, generator=loader_generator), expected_loader)
    metadata = inspect_checkpoint(path)
    assert metadata["prototype_available_count"] == 247


@pytest.mark.parametrize(
    "field,value",
    [
        ("outer_fold", 1),
        ("text_backend", "macbert"),
        ("input_channels", 127),
        ("asset_hashes", {"fold": "bad"}),
        ("subject_index_mapping", {"sub-02": 0}),
        ("keyword_index_mapping", {"bad": 0}),
    ],
)
def test_checkpoint_rejects_provenance_mismatch(tmp_path, field, value) -> None:
    eeg, text, optimizer, scheduler, scaler = _objects()
    config = {"backend": "bge_m3"}
    assets = {"fold": "d" * 64}
    subjects = {"sub-01": 0}
    keywords = {f"kw-{index}": index for index in range(247)}
    early = EarlyStopping()
    path = tmp_path / "last.pt"
    save_training_checkpoint(
        path,
        outer_fold=0,
        epoch=0,
        global_optimizer_step=0,
        eeg_encoder=eeg,
        text_projector=text,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        prototype_bank=_bank(text),
        resolved_config=config,
        asset_hashes=assets,
        subject_index_mapping=subjects,
        keyword_index_mapping=keywords,
        current_validation_metrics={},
        best_validation_metrics={},
        early_stopping=early,
    )
    arguments = dict(
        outer_fold=0,
        text_backend="bge_m3",
        input_channels=128,
        subject_index_mapping=subjects,
        keyword_index_mapping=keywords,
        asset_hashes=assets,
        config_sha256=canonical_sha256(config),
    )
    arguments[field] = value
    with pytest.raises(ValueError, match="mismatch"):
        load_training_checkpoint(
            path,
            expected=CheckpointExpectations(**arguments),
            eeg_encoder=eeg,
            text_projector=text,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            early_stopping=early,
        )
