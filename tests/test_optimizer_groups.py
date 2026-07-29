from __future__ import annotations

from pathlib import Path

import torch

from eeg_keyword_decoding.models import (
    build_context_text_projector,
    build_eeg_sequence_encoder,
)
from eeg_keyword_decoding.training import OptimizerConfig, build_adamw_optimizer


def _models():
    root = Path(__file__).resolve().parents[1]
    return (
        build_eeg_sequence_encoder(root / "configs/models/eeg_sequence_conv_v1.yaml"),
        build_context_text_projector(root / "configs/text/macbert_projection_v1.yaml"),
    )


def test_trainable_parameters_appear_exactly_once_with_explicit_lrs() -> None:
    eeg, text = _models()
    config = OptimizerConfig(
        eeg_lr=3e-4,
        text_projection_lr=1e-4,
        scalar_mix_lr=7e-5,
    )
    optimizer = build_adamw_optimizer(
        eeg_encoder=eeg, text_projector=text, config=config
    )
    parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    expected = [
        parameter
        for module in (eeg, text)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    assert len(parameters) == len({id(value) for value in parameters})
    assert {id(value) for value in parameters} == {id(value) for value in expected}
    scalar_group = next(
        group for group in optimizer.param_groups if group["group_name"].startswith("scalar_mix/")
    )
    assert scalar_group["lr"] == 7e-5


def test_bias_norm_and_scalar_mix_have_no_weight_decay_and_frozen_are_excluded() -> None:
    eeg, text = _models()
    frozen = text.projection.weight
    frozen.requires_grad_(False)
    optimizer = build_adamw_optimizer(
        eeg_encoder=eeg, text_projector=text, config=OptimizerConfig()
    )
    assert all(
        frozen is not parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    assert any(
        text.layer_logits is parameter and group["weight_decay"] == 0.0
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    assert any(
        text.projection.bias is parameter and group["weight_decay"] == 0.0
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    assert any(
        text.normalization.weight is parameter and group["weight_decay"] == 0.0
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
