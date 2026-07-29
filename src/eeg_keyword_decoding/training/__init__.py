"""Leakage-guarded single-fold training infrastructure."""

from .early_stopping import EarlyStopping, PRIMARY_VALIDATION_METRIC
from .checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointExpectations,
    LoadedTrainingState,
    canonical_sha256,
    inspect_checkpoint,
    load_training_checkpoint,
    save_training_checkpoint,
)
from .logging import JsonlLogger
from .epoch import TrainEpochConfig, TrainEpochResult, train_one_epoch
from .optimizer import OptimizerConfig, build_adamw_optimizer
from .outputs import create_unique_run_directory
from .prototype_refresh import PrototypeRefreshCoordinator, PrototypeRefreshResult
from .reproducibility import (
    ReproducibilityConfig,
    capture_rng_state,
    configure_reproducibility,
    make_dataloader_generator,
    restore_rng_state,
    seed_dataloader_worker,
)
from .scheduler import SchedulerConfig, build_scheduler
from .trainer import (
    FoldTrainer,
    FoldTrainerConfig,
    FoldTrainingResult,
    guard_fold_training_boundaries,
)

__all__ = [
    "EarlyStopping",
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointExpectations",
    "JsonlLogger",
    "LoadedTrainingState",
    "FoldTrainer",
    "FoldTrainerConfig",
    "FoldTrainingResult",
    "OptimizerConfig",
    "PRIMARY_VALIDATION_METRIC",
    "PrototypeRefreshCoordinator",
    "PrototypeRefreshResult",
    "ReproducibilityConfig",
    "SchedulerConfig",
    "TrainEpochConfig",
    "TrainEpochResult",
    "build_adamw_optimizer",
    "build_scheduler",
    "canonical_sha256",
    "capture_rng_state",
    "configure_reproducibility",
    "create_unique_run_directory",
    "guard_fold_training_boundaries",
    "inspect_checkpoint",
    "load_training_checkpoint",
    "make_dataloader_generator",
    "restore_rng_state",
    "save_training_checkpoint",
    "seed_dataloader_worker",
    "train_one_epoch",
]
