"""Optimal-transport alignment modules."""

from .context import ContextOTAligner
from .costs import masked_cosine_cost
from .outputs import (
    ContextOTOutput,
    CosineCostOutput,
    SinkhornOutput,
    TransportPoolOutput,
)
from .sinkhorn import (
    SINKHORN_CONFIG_SCHEMA,
    MaskedBalancedSinkhorn,
    MaskedBalancedSinkhornConfig,
)
from .transport import (
    TRANSPORT_POOL_CONFIG_SCHEMA,
    TransportPoolingConfig,
    transport_pool_eeg_to_words,
)

__all__ = [
    "ContextOTAligner",
    "ContextOTOutput",
    "CosineCostOutput",
    "MaskedBalancedSinkhorn",
    "MaskedBalancedSinkhornConfig",
    "SINKHORN_CONFIG_SCHEMA",
    "SinkhornOutput",
    "TRANSPORT_POOL_CONFIG_SCHEMA",
    "TransportPoolOutput",
    "TransportPoolingConfig",
    "masked_cosine_cost",
    "transport_pool_eeg_to_words",
]
