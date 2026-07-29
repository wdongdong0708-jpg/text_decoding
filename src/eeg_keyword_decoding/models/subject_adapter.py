from __future__ import annotations

import torch
from torch import nn


class SubjectFiLM(nn.Module):
    """Per-subject affine calibration over shared channel features."""

    def __init__(self, *, num_subjects: int, feature_dim: int) -> None:
        super().__init__()
        if num_subjects <= 0:
            raise ValueError("num_subjects must be positive")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        self.num_subjects = int(num_subjects)
        self.feature_dim = int(feature_dim)
        self.gamma = nn.Embedding(self.num_subjects, self.feature_dim)
        self.beta = nn.Embedding(self.num_subjects, self.feature_dim)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(
        self,
        sequence: torch.Tensor,
        subject_indices: torch.Tensor,
    ) -> torch.Tensor:
        if sequence.ndim != 3:
            raise ValueError(
                "SubjectFiLM sequence must have shape [batch, feature, time], "
                f"got {tuple(sequence.shape)}"
            )
        if sequence.shape[1] != self.feature_dim:
            raise ValueError(
                f"SubjectFiLM feature dim {sequence.shape[1]} does not match "
                f"configured {self.feature_dim}"
            )
        if subject_indices.shape != (sequence.shape[0],):
            raise ValueError(
                "subject_indices must have shape [batch], got "
                f"{tuple(subject_indices.shape)} for batch {sequence.shape[0]}"
            )
        if subject_indices.dtype != torch.int64:
            raise ValueError(
                f"subject_indices must be torch.int64, got "
                f"{subject_indices.dtype}"
            )
        if subject_indices.device != sequence.device:
            raise ValueError(
                "subject_indices and sequence must be on the same device"
            )
        if bool((subject_indices < 0).any()) or bool(
            (subject_indices >= self.num_subjects).any()
        ):
            raise IndexError(
                "subject index out of range [0, "
                f"{self.num_subjects}): {subject_indices.tolist()}"
            )
        gamma = self.gamma(subject_indices).unsqueeze(-1)
        beta = self.beta(subject_indices).unsqueeze(-1)
        return gamma * sequence + beta
