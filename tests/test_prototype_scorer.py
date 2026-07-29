from __future__ import annotations

import inspect

import torch
from torch.nn import functional as F

from eeg_keyword_decoding.evaluation import TextFreePrototypeScorer
from eeg_keyword_decoding.evaluation.prototype_scorer import UNAVAILABLE_SCORE
from eeg_keyword_decoding.prototypes.schema import PrototypeBank


def _bank(device: str = "cpu") -> PrototypeBank:
    generator = torch.Generator().manual_seed(11)
    vectors = F.normalize(torch.randn(247, 256, generator=generator), dim=-1)
    available = torch.ones(247, dtype=torch.bool)
    available[-1] = False
    vectors[-1].zero_()
    digest = "a" * 64
    return PrototypeBank(
        vectors=vectors.to(device),
        available_mask=available.to(device),
        keyword_ids=tuple(f"kw-{index}" for index in range(247)),
        train_sentence_df=torch.ones(247, dtype=torch.int64, device=device),
        train_group_df=torch.ones(247, dtype=torch.int64, device=device),
        outer_fold=0,
        text_backend="bge_m3",
        projector_state_hash=digest,
        source_cache_hash=digest,
        source_cache_metadata_hash=digest,
        fold_hash=digest,
        eligibility_hash=digest,
        lexical_mapping_hash=digest,
        metadata={},
    )


def test_scorer_interface_and_fixed_width() -> None:
    parameters = tuple(inspect.signature(TextFreePrototypeScorer.forward).parameters)
    assert parameters == ("self", "eeg_sequence", "eeg_mask", "prototype_bank")
    output = TextFreePrototypeScorer()(torch.randn(2, 5, 256), torch.ones(2, 5, dtype=torch.bool), _bank())
    assert output.scores.shape == (2, 247)
    assert output.available_mask.shape == (247,)
    assert torch.isfinite(output.scores).all()


def test_padding_values_and_batch_padding_do_not_change_scores() -> None:
    scorer = TextFreePrototypeScorer()
    bank = _bank()
    valid = torch.randn(1, 3, 256)
    alone = scorer(valid, torch.ones(1, 3, dtype=torch.bool), bank).scores
    padded = torch.cat((valid, torch.randn(1, 4, 256) * 1e8), dim=1)
    mask = torch.tensor([[True, True, True, False, False, False, False]])
    together = scorer(padded, mask, bank).scores
    assert torch.allclose(alone, together, atol=1e-6, rtol=1e-6)

    second = torch.randn(1, 7, 256)
    batch = torch.cat((padded, second), dim=0)
    batch_mask = torch.cat((mask, torch.ones(1, 7, dtype=torch.bool)), dim=0)
    batched = scorer(batch, batch_mask, bank).scores[0]
    assert torch.allclose(alone[0], batched, atol=1e-6, rtol=1e-6)


def test_repeating_identical_valid_state_does_not_add_length_bias() -> None:
    scorer = TextFreePrototypeScorer()
    bank = _bank()
    state = torch.randn(1, 1, 256)
    one = scorer(state, torch.ones(1, 1, dtype=torch.bool), bank).scores
    repeated = state.repeat(1, 9, 1)
    nine = scorer(repeated, torch.ones(1, 9, dtype=torch.bool), bank).scores
    assert torch.allclose(one, nine, atol=1e-6, rtol=1e-6)

    # This assertion detects a missing -log(T) normalization.
    raw_one = 0.07 * torch.logsumexp(
        F.normalize(state, dim=-1) @ bank.vectors.T / 0.07, dim=1
    )
    raw_nine = 0.07 * torch.logsumexp(
        F.normalize(repeated, dim=-1) @ bank.vectors.T / 0.07, dim=1
    )
    assert not torch.allclose(raw_one[:, :-1], raw_nine[:, :-1])


def test_padding_has_zero_gradient_and_unavailable_is_masked() -> None:
    scorer = TextFreePrototypeScorer()
    sequence = torch.randn(1, 4, 256, requires_grad=True)
    mask = torch.tensor([[True, True, False, False]])
    output = scorer(sequence, mask, _bank())
    output.scores[:, :-1].sum().backward()
    assert torch.count_nonzero(sequence.grad[:, 2:]) == 0
    assert not output.available_mask[-1]
    assert torch.all(output.scores[:, -1] == UNAVAILABLE_SCORE)


def test_scorer_fp32_and_amp_are_finite_and_differentiable() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sequence = torch.randn(2, 5, 256, device=device, requires_grad=True)
    mask = torch.ones(2, 5, dtype=torch.bool, device=device)
    scorer = TextFreePrototypeScorer().to(device)
    with torch.autocast(device_type=device, enabled=device == "cuda"):
        output = scorer(sequence, mask, _bank(device))
    assert output.scores.dtype == torch.float32
    assert torch.isfinite(output.scores).all()
    output.scores[:, :-1].mean().backward()
    assert sequence.grad is not None
    assert torch.isfinite(sequence.grad).all()
