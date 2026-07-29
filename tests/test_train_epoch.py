from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from eeg_keyword_decoding.training.epoch import TrainEpochConfig, train_one_epoch


class _Batch:
    context_words = torch.ones(1, 1, 1)
    roles = ("train",)
    eeg = torch.ones(1, 1, 1)
    eeg_mask = torch.ones(1, 1, dtype=torch.bool)
    subject_indices = torch.zeros(1, dtype=torch.int64)
    word_mask = torch.ones(1, 1, dtype=torch.bool)
    context_backend = "bge_m3"
    context_token_group_indices = torch.zeros(1, 1, dtype=torch.int64)
    surface_type_indices = torch.zeros(1, 1, dtype=torch.int64)
    sentence_group_indices = torch.zeros(1, dtype=torch.int64)
    word_keyword_indices = torch.zeros(1, 1, dtype=torch.int64)
    eeg_view_ids = ("view",)
    eeg_lengths = torch.ones(1, dtype=torch.int64)
    word_lengths = torch.ones(1, dtype=torch.int64)

    def to(self, device, non_blocking=True):
        del device, non_blocking
        return self


class _Loader:
    def __init__(self, count):
        self.count = count

    def __len__(self):
        return self.count

    def __iter__(self):
        return iter([_Batch() for _ in range(self.count)])


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, **kwargs):
        del kwargs
        value = self.weight.reshape(1, 1, 1)
        return SimpleNamespace(
            sequence=value,
            mask=torch.ones(1, 1, dtype=torch.bool),
        )


class _Aligner(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, *, eeg_sequence, **kwargs):
        del kwargs
        return eeg_sequence * self.weight


class _Loss(nn.Module):
    def __init__(self, *, nonfinite_loss=False, nonfinite_gradient=False):
        super().__init__()
        self.nonfinite_loss = nonfinite_loss
        self.nonfinite_gradient = nonfinite_gradient

    def forward(self, *, alignment, **kwargs):
        del kwargs
        if self.nonfinite_loss:
            total = alignment.sum() * torch.tensor(float("nan"))
        elif self.nonfinite_gradient:
            total = _FiniteForwardNanBackward.apply(alignment).sum()
        else:
            total = alignment.square().sum()
        return SimpleNamespace(
            total=total,
            ot_context=total,
            context_token=total * 0.5,
            prototype=total * 0.25,
            diagnostics={
                "plan_row_error_max": 0.0,
                "plan_column_error_max": 0.0,
                "token_valid_query_count": 1,
                "scalar_mix_weights": None,
            },
        )


class _FiniteForwardNanBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        return torch.full_like(gradient, float("nan"))


class _Bank:
    available_count = 1

    def to(self, device):
        del device
        return self


def _run(*, count=5, accumulation=2, loss=None):
    encoder = _Encoder()
    aligner = _Aligner()
    optimizer = torch.optim.SGD(
        [*encoder.parameters(), *aligner.parameters()], lr=0.01
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    result = train_one_epoch(
        epoch=0,
        global_optimizer_step=7,
        eeg_encoder=encoder,
        aligner=aligner,  # type: ignore[arg-type]
        loss_module=loss or _Loss(),  # type: ignore[arg-type]
        train_loader=_Loader(count),  # type: ignore[arg-type]
        prototype_bank=_Bank(),  # type: ignore[arg-type]
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=torch.amp.GradScaler("cpu", enabled=False),
        device="cpu",
        config=TrainEpochConfig(
            precision="fp32",
            gradient_accumulation_steps=accumulation,
        ),
    )
    return result, scheduler


def test_gradient_accumulation_and_scheduler_count_optimizer_steps() -> None:
    result, scheduler = _run(count=5, accumulation=2)
    assert result.micro_batches == 5
    assert result.optimizer_steps == 3
    assert result.global_optimizer_step == 10
    assert scheduler.last_epoch == 3


def test_nonfinite_loss_and_gradient_fail_immediately() -> None:
    with pytest.raises(FloatingPointError, match="Non-finite loss"):
        _run(count=1, loss=_Loss(nonfinite_loss=True))
    with pytest.raises(FloatingPointError, match="Non-finite gradient"):
        _run(count=1, loss=_Loss(nonfinite_gradient=True))
