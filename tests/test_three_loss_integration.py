from __future__ import annotations

import json

import pytest
import torch

from eeg_keyword_decoding.losses import (
    ContextOTThreeLoss,
    ThreeLossConfig,
)
from eeg_keyword_decoding.models import ContextTextProjector, TextProjectionConfig
from eeg_keyword_decoding.ot import ContextOTAligner, MaskedBalancedSinkhorn
from eeg_keyword_decoding.prototypes import PrototypeBank


def _projection_config(backend: str) -> TextProjectionConfig:
    macbert = backend == "macbert"
    return TextProjectionConfig.from_dict(
        {
            "schema_version": "context_text_projection_v1",
            "backend": backend,
            "cache_metadata_path": "unused",
            "input_dim": 8,
            "output_dim": 256,
            "normalization": "layer_norm",
            "projection": "linear",
            "scalar_mix": {
                "enabled": macbert,
                "layer_count": 4 if macbert else 0,
                "initialization": "zeros",
            },
            "expected_cache": {
                "model_id": "fake",
                "resolved_revision": "revision",
                "context_vectors_sha256": "a" * 64,
                "storage_dtype": "float32" if macbert else "float16",
                "layer_indices": [1, 2, 3, 4] if macbert else [],
            },
        }
    )


def _loss_config(**weights) -> ThreeLossConfig:
    return ThreeLossConfig.from_dict(
        {
            "schema_version": "context_ot_three_loss_v1",
            "loss": {
                "ot_context": {"weight": weights.get("ot", 1.0)},
                "context_token": {
                    "weight": weights.get("token", 1.0),
                    "temperature": 0.07,
                    "symmetric": True,
                    "reduction": "sample_balanced",
                },
                "prototype": {
                    "weight": weights.get("prototype", 0.5),
                    "temperature": 0.07,
                    "min_train_group_df": 10,
                    "reduction": "sample_balanced",
                },
            },
        }
    )


def _bank(device: torch.device) -> PrototypeBank:
    generator = torch.Generator().manual_seed(4)
    vectors = torch.zeros(247, 256)
    vectors[:10] = torch.nn.functional.normalize(
        torch.randn(10, 256, generator=generator),
        dim=-1,
    )
    available = torch.zeros(247, dtype=torch.bool)
    available[:10] = True
    bank = PrototypeBank(
        vectors=vectors,
        available_mask=available,
        keyword_ids=tuple(f"kw-{index}" for index in range(247)),
        train_sentence_df=torch.arange(247, dtype=torch.int64),
        train_group_df=torch.arange(247, dtype=torch.int64),
        outer_fold=0,
        text_backend="bge_m3",
        projector_state_hash="a" * 64,
        source_cache_hash="b" * 64,
        source_cache_metadata_hash="c" * 64,
        fold_hash="d" * 64,
        eligibility_hash="e" * 64,
        lexical_mapping_hash="f" * 64,
        metadata={"min_train_group_df": 10},
    )
    return bank.to(device)


def _forward(backend: str, device: torch.device):
    torch.manual_seed(3)
    projector = ContextTextProjector(_projection_config(backend)).to(device)
    aligner = ContextOTAligner(
        text_projector=projector,
        sinkhorn=MaskedBalancedSinkhorn(epsilon=0.1, iterations=30),
    ).to(device)
    eeg = torch.randn(2, 4, 256, device=device, requires_grad=True)
    eeg_mask = torch.tensor(
        [[True, True, True, False], [True, True, True, True]],
        device=device,
    )
    word_mask = torch.tensor(
        [[True, True, False], [True, True, True]],
        device=device,
    )
    if backend == "macbert":
        context_words = torch.randn(2, 3, 4, 8, device=device)
    else:
        context_words = torch.randn(2, 3, 8, device=device)
    alignment = aligner(
        eeg_sequence=eeg,
        eeg_mask=eeg_mask,
        context_words=context_words,
        word_mask=word_mask,
        context_backend=backend,
    )
    identities = {
        "word_mask": word_mask,
        "context_token_group_indices": torch.tensor(
            [[0, 1, -1], [2, 3, 4]],
            device=device,
        ),
        "surface_type_indices": torch.tensor(
            [[0, 1, -1], [2, 3, 4]],
            device=device,
        ),
        "sentence_group_indices": torch.tensor([0, 1], device=device),
        "word_keyword_indices": torch.tensor(
            [[0, -1, -1], [1, 2, -1]],
            device=device,
        ),
    }
    return eeg, projector, alignment, identities


@pytest.mark.parametrize("backend", ["macbert", "bge_m3"])
def test_three_loss_full_forward_backward_is_finite(backend):
    device = torch.device("cpu")
    eeg, projector, alignment, identities = _forward(backend, device)
    output = ContextOTThreeLoss(_loss_config())(
        alignment=alignment,
        prototype_bank=_bank(device),
        **identities,
    )
    assert torch.allclose(
        output.total,
        output.weighted_ot
        + output.weighted_token
        + output.weighted_prototype,
    )
    assert output.diagnostics["all_finite"] is True
    output.total.backward()
    assert eeg.grad is not None and bool(torch.isfinite(eeg.grad).all())
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in projector.parameters()
    )


def test_zero_weight_skips_token_and_prototype_and_all_zero_is_rejected():
    device = torch.device("cpu")
    _, _, alignment, identities = _forward("bge_m3", device)
    output = ContextOTThreeLoss(
        _loss_config(ot=1.0, token=0.0, prototype=0.0)
    )(
        alignment=alignment,
        prototype_bank=None,
        **identities,
    )
    assert output.context_token == 0
    assert output.prototype == 0
    assert output.diagnostics["zero_weight_skips"]["context_token"] is True
    assert output.diagnostics["zero_weight_skips"]["prototype"] is True

    with pytest.raises(ValueError, match="At least one"):
        _loss_config(ot=0.0, token=0.0, prototype=0.0)


def test_loss_config_metadata_is_json_serializable():
    config = _loss_config()
    restored = ThreeLossConfig.from_dict(
        json.loads(json.dumps(config.to_metadata()))
    )
    assert restored == config


def test_no_available_batch_target_keeps_ot_and_token_trainable():
    device = torch.device("cpu")
    _, _, alignment, identities = _forward("bge_m3", device)
    identities["word_keyword_indices"] = torch.full(
        (2, 3),
        -1,
        device=device,
        dtype=torch.int64,
    )
    output = ContextOTThreeLoss(_loss_config())(
        alignment=alignment,
        prototype_bank=_bank(device),
        **identities,
    )
    assert output.prototype == 0
    assert output.ot_context > 0
    assert output.context_token >= 0
    output.total.backward()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_amp_three_loss_keeps_sinkhorn_fp32():
    device = torch.device("cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        _, _, alignment, identities = _forward("bge_m3", device)
        output = ContextOTThreeLoss(_loss_config())(
            alignment=alignment,
            prototype_bank=_bank(device),
            **identities,
        )
    assert alignment.sinkhorn.plan.dtype == torch.float32
    assert output.total.dtype == torch.float32
    output.total.backward()
