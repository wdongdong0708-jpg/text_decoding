from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from eeg_keyword_decoding.losses import ContextOTThreeLoss, ThreeLossConfig
from eeg_keyword_decoding.models import ContextTextProjector, TextProjectionConfig
from eeg_keyword_decoding.ot import ContextOTAligner, MaskedBalancedSinkhorn
from eeg_keyword_decoding.prototypes import PrototypeBank


@dataclass(frozen=True)
class SyntheticOverfitResult:
    steps: int
    first_window_mean_loss: float
    final_window_mean_loss: float
    initial_prototype_accuracy: float
    final_prototype_accuracy: float
    maximum_gradient: float
    padding_loss_absolute_error: float
    resumed_next_loss_absolute_error: float
    resumed_parameter_max_absolute_error: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


class _SyntheticEEG(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.states = nn.Parameter(torch.randn(4, 2, 256) * 0.1)

    def forward(self) -> torch.Tensor:
        return self.states


def _projector() -> ContextTextProjector:
    return ContextTextProjector(
        TextProjectionConfig.from_dict(
            {
                "schema_version": "context_text_projection_v1",
                "backend": "bge_m3",
                "cache_metadata_path": "unused",
                "input_dim": 8,
                "output_dim": 256,
                "normalization": "layer_norm",
                "projection": "linear",
                "scalar_mix": {
                    "enabled": False,
                    "layer_count": 0,
                    "initialization": "zeros",
                },
                "expected_cache": {
                    "model_id": "synthetic",
                    "resolved_revision": "fixed",
                    "context_vectors_sha256": "a" * 64,
                    "storage_dtype": "float16",
                    "layer_indices": [],
                },
            }
        )
    )


def _loss_module() -> ContextOTThreeLoss:
    return ContextOTThreeLoss(
        ThreeLossConfig.from_dict(
            {
                "schema_version": "context_ot_three_loss_v1",
                "loss": {
                    "ot_context": {"weight": 1.0},
                    "context_token": {
                        "weight": 1.0,
                        "temperature": 0.1,
                        "symmetric": True,
                        "reduction": "sample_balanced",
                    },
                    "prototype": {
                        "weight": 0.5,
                        "temperature": 0.1,
                        "min_train_group_df": 10,
                        "reduction": "sample_balanced",
                    },
                },
            }
        )
    )


def _bank() -> PrototypeBank:
    generator = torch.Generator().manual_seed(909)
    vectors = torch.zeros(247, 256)
    vectors[:2] = F.normalize(torch.randn(2, 256, generator=generator), dim=-1)
    available = torch.zeros(247, dtype=torch.bool)
    available[:2] = True
    digest = "b" * 64
    return PrototypeBank(
        vectors=vectors,
        available_mask=available,
        keyword_ids=tuple(f"kw-{index}" for index in range(247)),
        train_sentence_df=torch.tensor([20, 20] + [0] * 245),
        train_group_df=torch.tensor([20, 20] + [0] * 245),
        outer_fold=0,
        text_backend="bge_m3",
        projector_state_hash=digest,
        source_cache_hash=digest,
        source_cache_metadata_hash=digest,
        fold_hash=digest,
        eligibility_hash=digest,
        lexical_mapping_hash=digest,
        metadata={"min_train_group_df": 10},
    )


def run_synthetic_tiny_overfit(
    *,
    steps: int = 40,
    seed: int = 42,
) -> SyntheticOverfitResult:
    if steps < 10:
        raise ValueError("Synthetic overfit needs at least 10 steps")
    torch.manual_seed(seed)
    eeg = _SyntheticEEG()
    projector = _projector()
    aligner = ContextOTAligner(
        text_projector=projector,
        sinkhorn=MaskedBalancedSinkhorn(epsilon=0.1, iterations=20),
    )
    loss_module = _loss_module()
    bank = _bank()
    optimizer = torch.optim.AdamW(
        [*eeg.parameters(), *projector.parameters()],
        lr=1e-2,
        weight_decay=0.0,
    )
    context = torch.zeros(4, 2, 8)
    context[:, 0, 0] = 2.0
    context[:, 1, 1] = 2.0
    context += torch.randn_like(context) * 0.01
    eeg_mask = torch.ones(4, 2, dtype=torch.bool)
    word_mask = torch.ones(4, 2, dtype=torch.bool)
    token_groups = torch.tensor([[0, 1]] * 4)
    surfaces = torch.tensor([[0, 1]] * 4)
    sentence_groups = torch.arange(4)
    keyword_indices = torch.tensor([[0, 1]] * 4)

    def forward(states: torch.Tensor, mask: torch.Tensor):
        alignment = aligner(
            eeg_sequence=states,
            eeg_mask=mask,
            context_words=context,
            word_mask=word_mask,
            context_backend="bge_m3",
        )
        output = loss_module(
            alignment=alignment,
            word_mask=word_mask,
            context_token_group_indices=token_groups,
            surface_type_indices=surfaces,
            sentence_group_indices=sentence_groups,
            word_keyword_indices=keyword_indices,
            prototype_bank=bank,
        )
        return alignment, output

    def accuracy() -> float:
        with torch.no_grad():
            alignment, _ = forward(eeg(), eeg_mask)
            similarities = F.normalize(
                alignment.word_conditioned_eeg.sequence, dim=-1
            ) @ bank.vectors[:2].T
            targets = keyword_indices
            return float((similarities.argmax(dim=-1) == targets).float().mean())

    initial_accuracy = accuracy()
    losses: list[float] = []
    maximum_gradient = 0.0
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        _, output = forward(eeg(), eeg_mask)
        output.total.backward()
        gradients = [
            parameter.grad
            for parameter in (*eeg.parameters(), *projector.parameters())
            if parameter.grad is not None
        ]
        if not gradients or not all(bool(torch.isfinite(value).all()) for value in gradients):
            raise FloatingPointError("Synthetic overfit produced a non-finite gradient")
        maximum_gradient = max(
            maximum_gradient,
            max(float(value.detach().abs().max()) for value in gradients),
        )
        optimizer.step()
        losses.append(float(output.total.detach()))
    final_accuracy = accuracy()

    with torch.no_grad():
        _, plain = forward(eeg(), eeg_mask)
        padded_states = torch.cat((eeg(), torch.randn(4, 3, 256) * 1e6), dim=1)
        padded_mask = torch.cat(
            (eeg_mask, torch.zeros(4, 3, dtype=torch.bool)), dim=1
        )
        _, padded = forward(padded_states, padded_mask)
        padding_error = float((plain.total - padded.total).abs())

    model_state = copy.deepcopy(eeg.state_dict())
    projector_state = copy.deepcopy(projector.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())

    def resume_step() -> tuple[float, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        optimizer.zero_grad(set_to_none=True)
        _, output = forward(eeg(), eeg_mask)
        output.total.backward()
        optimizer.step()
        return (
            float(output.total.detach()),
            copy.deepcopy(eeg.state_dict()),
            copy.deepcopy(projector.state_dict()),
        )

    first_loss, first_eeg, first_projector = resume_step()
    eeg.load_state_dict(model_state)
    projector.load_state_dict(projector_state)
    optimizer.load_state_dict(optimizer_state)
    second_loss, second_eeg, second_projector = resume_step()
    parameter_error = max(
        float((first_eeg[name] - second_eeg[name]).abs().max())
        for name in first_eeg
    )
    parameter_error = max(
        parameter_error,
        max(
            float((first_projector[name] - second_projector[name]).abs().max())
            for name in first_projector
        ),
    )
    window = min(5, steps // 2)
    return SyntheticOverfitResult(
        steps=steps,
        first_window_mean_loss=sum(losses[:window]) / window,
        final_window_mean_loss=sum(losses[-window:]) / window,
        initial_prototype_accuracy=initial_accuracy,
        final_prototype_accuracy=final_accuracy,
        maximum_gradient=maximum_gradient,
        padding_loss_absolute_error=padding_error,
        resumed_next_loss_absolute_error=abs(first_loss - second_loss),
        resumed_parameter_max_absolute_error=parameter_error,
    )
