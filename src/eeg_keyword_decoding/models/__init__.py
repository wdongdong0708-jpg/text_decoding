"""EEG sequence models."""

from .eeg_sequence import (
    EEG_SEQUENCE_CONFIG_SCHEMA,
    EEGSequenceEncoder,
    EEGSequenceEncoderConfig,
    EEGSequenceOutput,
    SubjectAdapterConfig,
    build_eeg_sequence_encoder,
)
from .masked_ops import (
    conv1d_output_lengths,
    lengths_from_prefix_mask,
    lengths_to_mask,
)
from .subject_adapter import SubjectFiLM

__all__ = [
    "EEG_SEQUENCE_CONFIG_SCHEMA",
    "EEGSequenceEncoder",
    "EEGSequenceEncoderConfig",
    "EEGSequenceOutput",
    "SubjectAdapterConfig",
    "SubjectFiLM",
    "build_eeg_sequence_encoder",
    "conv1d_output_lengths",
    "lengths_from_prefix_mask",
    "lengths_to_mask",
]
