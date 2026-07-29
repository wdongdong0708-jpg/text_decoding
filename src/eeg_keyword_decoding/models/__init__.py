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
from .text_projection import (
    TEXT_PROJECTION_CONFIG_SCHEMA,
    ContextTextProjector,
    TextBackend,
    TextProjectionConfig,
    TextProjectionOutput,
    build_context_text_projector,
)

__all__ = [
    "EEG_SEQUENCE_CONFIG_SCHEMA",
    "EEGSequenceEncoder",
    "EEGSequenceEncoderConfig",
    "EEGSequenceOutput",
    "SubjectAdapterConfig",
    "SubjectFiLM",
    "TEXT_PROJECTION_CONFIG_SCHEMA",
    "ContextTextProjector",
    "TextBackend",
    "TextProjectionConfig",
    "TextProjectionOutput",
    "build_eeg_sequence_encoder",
    "build_context_text_projector",
    "conv1d_output_lengths",
    "lengths_from_prefix_mask",
    "lengths_to_mask",
]
