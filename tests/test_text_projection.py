from __future__ import annotations

import copy
import io
import json
from pathlib import Path

import pytest
import torch

from eeg_keyword_decoding.models import (
    ContextTextProjector,
    TextProjectionConfig,
    build_context_text_projector,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def projection_config(backend: str) -> TextProjectionConfig:
    if backend == "macbert":
        return TextProjectionConfig.from_dict(
            {
                "schema_version": "context_text_projection_v1",
                "backend": "macbert",
                "cache_metadata_path": "unused",
                "input_dim": 6,
                "output_dim": 256,
                "normalization": "layer_norm",
                "projection": "linear",
                "scalar_mix": {
                    "enabled": True,
                    "layer_count": 4,
                    "initialization": "zeros",
                },
                "expected_cache": {
                    "model_id": "fake",
                    "resolved_revision": "revision",
                    "context_vectors_sha256": "a" * 64,
                    "storage_dtype": "float32",
                    "layer_indices": [1, 2, 3, 4],
                },
            }
        )
    return TextProjectionConfig.from_dict(
        {
            "schema_version": "context_text_projection_v1",
            "backend": "bge_m3",
            "cache_metadata_path": "unused",
            "input_dim": 7,
            "output_dim": 256,
            "normalization": "layer_norm",
            "projection": "linear",
            "scalar_mix": {
                "enabled": False,
                "layer_count": 0,
                "initialization": "zeros",
            },
            "expected_cache": {
                "model_id": "fake",
                "resolved_revision": "revision",
                "context_vectors_sha256": "b" * 64,
                "storage_dtype": "float16",
                "layer_indices": [],
            },
        }
    )


def prefix_mask(lengths: list[int], maximum: int) -> torch.Tensor:
    return (
        torch.arange(maximum).unsqueeze(0)
        < torch.tensor(lengths).unsqueeze(1)
    )


@pytest.mark.parametrize(
    ("backend", "input_shape"),
    [
        ("macbert", (2, 5, 4, 6)),
        ("bge_m3", (2, 5, 7)),
    ],
)
def test_text_projectors_share_256d_output_contract(backend, input_shape):
    model = ContextTextProjector(projection_config(backend))
    output = model(
        context_words=torch.randn(*input_shape),
        word_mask=prefix_mask([3, 5], 5),
        backend=backend,
    )
    assert output.sequence.shape == (2, 5, 256)
    assert output.mask.shape == (2, 5)
    assert torch.count_nonzero(output.sequence[0, 3:]) == 0
    assert (output.scalar_mix_weights is not None) == (
        backend == "macbert"
    )


def test_macbert_scalar_mix_initializes_uniform_and_sums_to_one():
    model = ContextTextProjector(projection_config("macbert"))
    output = model(
        context_words=torch.randn(1, 2, 4, 6),
        word_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    torch.testing.assert_close(
        output.scalar_mix_weights,
        torch.full((4,), 0.25),
    )
    torch.testing.assert_close(
        output.scalar_mix_weights.sum(),
        torch.tensor(1.0),
    )


def test_macbert_scalar_mix_receives_finite_gradient():
    model = ContextTextProjector(projection_config("macbert"))
    layers = torch.randn(2, 3, 4, 6)
    layers[:, :, 1] += 3.0
    output = model(
        context_words=layers,
        word_mask=prefix_mask([2, 3], 3),
    )
    output.sequence.square().mean().backward()
    assert model.layer_logits is not None
    assert model.layer_logits.grad is not None
    assert torch.isfinite(model.layer_logits.grad).all()


@pytest.mark.parametrize("backend", ["macbert", "bge_m3"])
def test_text_projection_extra_padding_does_not_change_valid_words(backend):
    torch.manual_seed(4)
    model = ContextTextProjector(projection_config(backend)).eval()
    tail = (4, 6) if backend == "macbert" else (7,)
    original = torch.randn(1, 2, *tail)
    baseline = model(
        context_words=original,
        word_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    padded = torch.randn(1, 5, *tail) * 100_000
    padded[:, :2] = original
    observed = model(
        context_words=padded,
        word_mask=prefix_mask([2], 5),
    )
    torch.testing.assert_close(
        observed.sequence[:, :2],
        baseline.sequence,
        rtol=1e-5,
        atol=1e-6,
    )
    assert torch.count_nonzero(observed.sequence[:, 2:]) == 0


@pytest.mark.parametrize("backend", ["macbert", "bge_m3"])
def test_backend_and_input_shape_mismatch_are_rejected(backend):
    model = ContextTextProjector(projection_config(backend))
    wrong = (
        torch.randn(1, 2, 7)
        if backend == "macbert"
        else torch.randn(1, 2, 4, 6)
    )
    with pytest.raises(ValueError, match="must have shape"):
        model(
            context_words=wrong,
            word_mask=torch.ones(1, 2, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="Batch backend"):
        good = (
            torch.randn(1, 2, 4, 6)
            if backend == "macbert"
            else torch.randn(1, 2, 7)
        )
        model(
            context_words=good,
            word_mask=torch.ones(1, 2, dtype=torch.bool),
            backend="bge_m3" if backend == "macbert" else "macbert",
        )


def test_text_projection_rejects_non_finite_input():
    model = ContextTextProjector(projection_config("bge_m3"))
    value = torch.randn(1, 2, 7)
    value[0, 1, 0] = torch.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        model(
            context_words=value,
            word_mask=torch.ones(1, 2, dtype=torch.bool),
        )


def test_text_projection_state_dict_round_trip_is_exact():
    config = projection_config("macbert")
    first = ContextTextProjector(config).eval()
    value = torch.randn(2, 3, 4, 6)
    mask = prefix_mask([2, 3], 3)
    expected = first(context_words=value, word_mask=mask)
    buffer = io.BytesIO()
    torch.save(first.state_dict(), buffer)
    buffer.seek(0)
    second = ContextTextProjector(copy.deepcopy(config)).eval()
    second.load_state_dict(torch.load(buffer, weights_only=True))
    observed = second(context_words=value, word_mask=mask)
    torch.testing.assert_close(observed.sequence, expected.sequence)
    torch.testing.assert_close(
        observed.scalar_mix_weights,
        expected.scalar_mix_weights,
    )


def test_real_projection_configs_validate_cache_metadata():
    macbert = build_context_text_projector(
        PROJECT_ROOT / "configs" / "text" / "macbert_projection_v1.yaml"
    )
    bge = build_context_text_projector(
        PROJECT_ROOT / "configs" / "text" / "bge_m3_projection_v1.yaml"
    )
    assert macbert.config.input_dim == 768
    assert bge.config.input_dim == 1024
    assert macbert.total_parameter_count == 198_404
    assert bge.total_parameter_count == 264_448


def test_bge_metadata_rejects_dense_sentence_representation():
    config = TextProjectionConfig.from_yaml(
        PROJECT_ROOT / "configs" / "text" / "bge_m3_projection_v1.yaml"
    )
    metadata_path = (
        PROJECT_ROOT
        / "data"
        / "cache"
        / "context_words"
        / "bge_m3_colbert_v1"
        / "metadata.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["extraction"]["dense_sentence_vectors_used"] = True
    with pytest.raises(ValueError, match="dense sentence vectors"):
        config.validate_cache_metadata(metadata)
