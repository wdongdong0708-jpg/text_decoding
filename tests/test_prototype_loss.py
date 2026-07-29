from __future__ import annotations

import torch

from eeg_keyword_decoding.losses import prototype_classification_loss
from eeg_keyword_decoding.prototypes import PrototypeBank


def _bank() -> PrototypeBank:
    vectors = torch.zeros(247, 256)
    vectors[0, 0] = 1
    vectors[1, 1] = 1
    vectors[2, 2] = 1
    available = torch.zeros(247, dtype=torch.bool)
    available[:3] = True
    bank = PrototypeBank(
        vectors=vectors,
        available_mask=available,
        keyword_ids=tuple(f"kw-{index}" for index in range(247)),
        train_sentence_df=torch.ones(247, dtype=torch.int64),
        train_group_df=torch.ones(247, dtype=torch.int64),
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
    bank.validate()
    return bank


def test_correct_prototype_similarity_reduces_loss():
    bank = _bank()
    correct = torch.zeros(1, 1, 256)
    correct[0, 0, 0] = 1
    wrong = torch.zeros(1, 1, 256)
    wrong[0, 0, 1] = 1
    kwargs = {
        "word_keyword_indices": torch.tensor([[0]]),
        "word_mask": torch.tensor([[True]]),
        "prototype_bank": bank,
        "temperature": 0.1,
    }
    assert prototype_classification_loss(
        word_conditioned_eeg=correct,
        **kwargs,
    ).loss < prototype_classification_loss(
        word_conditioned_eeg=wrong,
        **kwargs,
    ).loss


def test_non_master_and_unavailable_targets_are_ignored_and_reported():
    bank = _bank()
    eeg = torch.randn(1, 3, 256)
    output = prototype_classification_loss(
        word_conditioned_eeg=eeg,
        word_keyword_indices=torch.tensor([[-1, 5, 0]]),
        word_mask=torch.ones(1, 3, dtype=torch.bool),
        prototype_bank=bank,
    )
    assert output.valid_word_count == 1
    assert output.ignored_non_master_count == 1
    assert output.ignored_unavailable_target_count == 1


def test_unavailable_prototypes_are_excluded_from_denominator():
    bank = _bank()
    eeg = torch.zeros(1, 1, 256)
    output = prototype_classification_loss(
        word_conditioned_eeg=eeg,
        word_keyword_indices=torch.tensor([[0]]),
        word_mask=torch.tensor([[True]]),
        prototype_bank=bank,
        temperature=0.1,
    )
    assert torch.allclose(output.loss, torch.log(torch.tensor(3.0)))
    assert output.available_prototype_count == 3


def test_empty_valid_keyword_batch_returns_connected_finite_zero():
    bank = _bank()
    eeg = torch.randn(2, 2, 256, requires_grad=True)
    output = prototype_classification_loss(
        word_conditioned_eeg=eeg,
        word_keyword_indices=torch.full((2, 2), -1),
        word_mask=torch.ones(2, 2, dtype=torch.bool),
        prototype_bank=bank,
    )
    assert output.loss == 0
    assert output.valid_word_count == 0
    output.loss.backward()
    assert eeg.grad is not None
    assert bool(torch.isfinite(eeg.grad).all())


def test_prototype_loss_is_sample_balanced_and_bank_is_detached():
    bank = _bank()
    bank.vectors.requires_grad_(True)
    eeg = torch.zeros(2, 2, 256, requires_grad=True)
    eeg.data[0, 0, 0] = 1
    eeg.data[0, 1, 1] = 1
    eeg.data[1, 0, 2] = 1
    output = prototype_classification_loss(
        word_conditioned_eeg=eeg,
        word_keyword_indices=torch.tensor([[0, 1], [2, -1]]),
        word_mask=torch.tensor([[True, True], [True, False]]),
        prototype_bank=bank,
    )
    assert output.valid_words_per_sample.tolist() == [2, 1]
    assert torch.allclose(output.loss, output.sample_loss.mean())
    output.loss.backward()
    assert eeg.grad is not None
    assert bank.vectors.grad is None
