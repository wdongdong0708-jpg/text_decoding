from __future__ import annotations

import pytest
import torch

from eeg_keyword_decoding.losses import context_token_info_nce


def _call(
    eeg: torch.Tensor,
    text: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    context: torch.Tensor | None = None,
    surface: torch.Tensor | None = None,
    sentence_groups: torch.Tensor | None = None,
    temperature: float = 0.1,
):
    batch, words, _ = eeg.shape
    return context_token_info_nce(
        word_conditioned_eeg=eeg,
        projected_context_words=text,
        word_mask=(
            mask
            if mask is not None
            else torch.ones(batch, words, dtype=torch.bool)
        ),
        context_token_group_indices=(
            context
            if context is not None
            else torch.arange(batch * words).reshape(batch, words)
        ),
        surface_type_indices=(
            surface
            if surface is not None
            else torch.arange(batch * words).reshape(batch, words)
        ),
        sentence_group_indices=(
            sentence_groups
            if sentence_groups is not None
            else torch.arange(batch)
        ),
        temperature=temperature,
        symmetric=True,
    )


def test_paired_context_tokens_score_better_than_shuffled():
    text = torch.eye(4).reshape(2, 2, 4)
    aligned = _call(text.clone(), text)
    shuffled = _call(text.flip(0), text)
    assert aligned.loss < shuffled.loss


def test_same_context_across_views_is_multi_positive():
    text = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]], [[0.0, 1.0]]])
    output = _call(
        text.clone(),
        text,
        context=torch.tensor([[10], [10], [20]]),
        surface=torch.tensor([[1], [1], [2]]),
        sentence_groups=torch.tensor([3, 3, 4]),
    )
    assert output.positive_counts.tolist() == [2, 2, 1]
    assert output.valid_query_count == 3


def test_same_lexical_type_different_context_is_neither_positive_nor_negative():
    text = torch.tensor([[[1.0, 0.0]], [[0.8, 0.2]], [[0.0, 1.0]]])
    output = _call(
        text.clone(),
        text,
        context=torch.tensor([[10], [11], [12]]),
        surface=torch.tensor([[1], [1], [2]]),
    )
    assert output.positive_counts.tolist() == [1, 1, 1]
    assert output.negative_counts.tolist() == [1, 1, 2]
    assert output.false_negative_mask_count == 2


def test_no_valid_negative_has_defined_zero_contribution():
    value = torch.randn(2, 1, 4, requires_grad=True)
    output = _call(
        value,
        value,
        context=torch.tensor([[1], [2]]),
        surface=torch.tensor([[7], [7]]),
    )
    assert output.loss == 0
    assert output.contributing_query_count == 0
    output.loss.backward()
    assert value.grad is not None
    assert bool(torch.isfinite(value.grad).all())


def test_padding_and_batch_permutation_do_not_change_loss():
    text = torch.eye(4).reshape(2, 2, 4)
    base = _call(text.clone(), text)
    padded_eeg = torch.cat([text, torch.randn(2, 2, 4)], dim=1)
    padded_text = torch.cat([text, torch.randn(2, 2, 4)], dim=1)
    padded = _call(
        padded_eeg,
        padded_text,
        mask=torch.tensor([[True, True, False, False]] * 2),
        context=torch.tensor([[0, 1, -1, -1], [2, 3, -1, -1]]),
        surface=torch.tensor([[0, 1, -1, -1], [2, 3, -1, -1]]),
    )
    assert torch.allclose(base.loss, padded.loss)

    permutation = torch.tensor([1, 0])
    permuted = _call(text[permutation], text[permutation])
    assert torch.allclose(base.loss, permuted.loss)


def test_sample_balanced_reduction_and_symmetric_average():
    text = torch.eye(4).reshape(2, 2, 4)
    output = _call(text.clone(), text)
    assert torch.allclose(
        output.eeg_to_text_loss,
        output.eeg_to_text_sample_loss.mean(),
    )
    assert torch.allclose(
        output.loss,
        0.5 * (output.eeg_to_text_loss + output.text_to_eeg_loss),
    )
    assert output.valid_words_per_sample.tolist() == [2, 2]


def test_sample_balanced_reduction_does_not_overweight_long_sentence():
    eeg = torch.randn(2, 4, 6)
    text = torch.randn(2, 4, 6)
    mask = torch.tensor(
        [[True, False, False, False], [True, True, True, True]]
    )
    identity = torch.tensor(
        [[0, -1, -1, -1], [1, 2, 3, 4]]
    )
    output = _call(
        eeg,
        text,
        mask=mask,
        context=identity,
        surface=identity,
    )
    assert output.valid_words_per_sample.tolist() == [1, 4]
    assert torch.allclose(
        output.eeg_to_text_loss,
        output.eeg_to_text_sample_loss.mean(),
    )


def test_context_token_gradients_reach_both_modalities():
    eeg = torch.randn(2, 2, 5, requires_grad=True)
    text = torch.randn(2, 2, 5, requires_grad=True)
    output = _call(eeg, text)
    output.loss.backward()
    assert eeg.grad is not None and bool(torch.isfinite(eeg.grad).all())
    assert text.grad is not None and bool(torch.isfinite(text.grad).all())


def test_context_token_rejects_nonpositive_temperature():
    value = torch.randn(1, 1, 3)
    with pytest.raises(ValueError, match="temperature"):
        _call(value, value, temperature=0.0)
