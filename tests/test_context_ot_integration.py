from __future__ import annotations

import torch

from eeg_keyword_decoding.models import (
    ContextTextProjector,
    TextProjectionConfig,
)
from eeg_keyword_decoding.ot import ContextOTAligner, MaskedBalancedSinkhorn


def _config() -> TextProjectionConfig:
    return TextProjectionConfig.from_dict(
        {
            "schema_version": "context_text_projection_v1",
            "backend": "macbert",
            "cache_metadata_path": "unused",
            "input_dim": 8,
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


def test_context_ot_chain_shapes_masks_and_gradients():
    torch.manual_seed(12)
    projector = ContextTextProjector(_config())
    aligner = ContextOTAligner(
        text_projector=projector,
        sinkhorn=MaskedBalancedSinkhorn(iterations=75),
    )
    eeg = torch.randn(2, 7, 256, requires_grad=True)
    context = torch.randn(2, 4, 4, 8, requires_grad=True)
    eeg_mask = torch.tensor(
        [[1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1]],
        dtype=torch.bool,
    )
    word_mask = torch.tensor(
        [[1, 1, 0, 0], [1, 1, 1, 1]],
        dtype=torch.bool,
    )
    output = aligner(
        eeg_sequence=eeg,
        eeg_mask=eeg_mask,
        context_words=context,
        word_mask=word_mask,
        context_backend="macbert",
    )
    assert output.text.sequence.shape == (2, 4, 256)
    assert output.cost.cost.shape == (2, 7, 4)
    assert output.sinkhorn.plan.shape == (2, 7, 4)
    assert output.word_conditioned_eeg.sequence.shape == (2, 4, 256)
    assert torch.count_nonzero(
        output.sinkhorn.plan[~output.cost.valid_pair_mask]
    ) == 0
    assert torch.count_nonzero(
        output.word_conditioned_eeg.sequence[~word_mask]
    ) == 0
    output.sinkhorn.transport_cost.mean().backward()
    assert eeg.grad is not None and torch.isfinite(eeg.grad).all()
    assert context.grad is not None and torch.isfinite(context.grad).all()
    assert projector.projection.weight.grad is not None
    assert projector.layer_logits is not None
    assert projector.layer_logits.grad is not None
    assert torch.isfinite(projector.layer_logits.grad).all()
    assert torch.count_nonzero(eeg.grad[0, 4:]) == 0
    assert torch.count_nonzero(context.grad[0, 2:]) == 0


def test_context_ot_repeated_run_is_deterministic_in_eval_mode():
    torch.manual_seed(13)
    aligner = ContextOTAligner(
        text_projector=ContextTextProjector(_config()),
        sinkhorn=MaskedBalancedSinkhorn(iterations=50),
    ).eval()
    inputs = {
        "eeg_sequence": torch.randn(1, 5, 256),
        "eeg_mask": torch.ones(1, 5, dtype=torch.bool),
        "context_words": torch.randn(1, 3, 4, 8),
        "word_mask": torch.ones(1, 3, dtype=torch.bool),
        "context_backend": "macbert",
    }
    first = aligner(**inputs)
    second = aligner(**inputs)
    torch.testing.assert_close(
        second.sinkhorn.plan,
        first.sinkhorn.plan,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        second.word_conditioned_eeg.sequence,
        first.word_conditioned_eeg.sequence,
        rtol=0,
        atol=0,
    )
