from __future__ import annotations

import inspect
from dataclasses import dataclass

import pytest

from eeg_keyword_decoding.training import FoldTrainer, guard_fold_training_boundaries


@dataclass(frozen=True)
class _Record:
    text_embedding_idx: int


@dataclass(frozen=True)
class _Assignment:
    sentence_group_id: str


class _Split:
    def __init__(self, groups):
        self.groups = groups

    def assignment(self, outer_fold, text_embedding_idx):
        del outer_fold
        return _Assignment(self.groups[text_embedding_idx])


class _Dataset:
    def __init__(
        self,
        *,
        role,
        fold=0,
        include_context_targets=None,
        indices=(1, 2),
        groups=None,
        context_store_root=None,
    ):
        self.role = role
        self.outer_fold = fold
        self.include_context_targets = (
            role == "train"
            if include_context_targets is None
            else include_context_targets
        )
        self.context_store_root = context_store_root
        self.records = tuple(_Record(index) for index in indices)
        self.split_index = _Split(
            groups or {index: f"group-{index}" for index in indices}
        )


def test_fold_trainer_api_has_no_test_dataset_or_loader() -> None:
    parameters = inspect.signature(FoldTrainer.__init__).parameters
    assert "test_dataset" not in parameters
    assert "test_loader" not in parameters


def test_role_and_context_target_guards() -> None:
    train = _Dataset(role="train", indices=(1, 2))
    validation = _Dataset(role="validation", indices=(3, 4))
    guard_fold_training_boundaries(train, validation)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="role=train"):
        guard_fold_training_boundaries(
            _Dataset(role="test", indices=(1, 2)), validation  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="role=validation"):
        guard_fold_training_boundaries(
            train, _Dataset(role="test", indices=(3, 4))  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="must not include"):
        guard_fold_training_boundaries(
            train,
            _Dataset(
                role="validation",
                include_context_targets=True,
                indices=(3, 4),
            ),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="text cache"):
        guard_fold_training_boundaries(
            train,
            _Dataset(
                role="validation",
                indices=(3, 4),
                context_store_root="forbidden",
            ),  # type: ignore[arg-type]
        )


def test_fold_text_and_context_group_overlap_are_rejected() -> None:
    with pytest.raises(ValueError, match="text_embedding_idx overlap"):
        guard_fold_training_boundaries(
            _Dataset(role="train", indices=(1, 2)),
            _Dataset(role="validation", indices=(2, 3)),
        )  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sentence_group_id overlap"):
        guard_fold_training_boundaries(
            _Dataset(
                role="train",
                indices=(1,),
                groups={1: "duplicate"},
            ),
            _Dataset(
                role="validation",
                indices=(2,),
                groups={2: "duplicate"},
            ),
        )  # type: ignore[arg-type]


def test_fold_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="outer folds differ"):
        guard_fold_training_boundaries(
            _Dataset(role="train", fold=0, indices=(1,)),
            _Dataset(role="validation", fold=1, indices=(2,)),
        )  # type: ignore[arg-type]
