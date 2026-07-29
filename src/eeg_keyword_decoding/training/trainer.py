from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from eeg_keyword_decoding.data import ContextEEGDataset
from eeg_keyword_decoding.evaluation import (
    FoldKeywordEligibility,
    TextFreePrototypeScorer,
)
from eeg_keyword_decoding.evaluation.validation import (
    TextFreeValidationResult,
    run_text_free_validation,
)
from eeg_keyword_decoding.losses import ContextOTThreeLoss
from eeg_keyword_decoding.models import ContextTextProjector
from eeg_keyword_decoding.ot import ContextOTAligner
from eeg_keyword_decoding.prototypes import PrototypeBank, save_prototype_bank

from .checkpoint import (
    CheckpointExpectations,
    canonical_sha256,
    load_training_checkpoint,
    save_training_checkpoint,
)
from .early_stopping import EarlyStopping, PRIMARY_VALIDATION_METRIC
from .epoch import TrainEpochConfig, TrainEpochResult, train_one_epoch
from .logging import JsonlLogger
from .prototype_refresh import PrototypeRefreshCoordinator


@dataclass(frozen=True)
class FoldTrainerConfig:
    epochs: int
    train_epoch: TrainEpochConfig
    max_validation_batches: int | None = None
    smoke_run: bool = False
    not_for_scientific_reporting: bool = False

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.smoke_run and not self.not_for_scientific_reporting:
            raise ValueError("Smoke outputs must be marked not_for_scientific_reporting")


@dataclass(frozen=True)
class FoldTrainingResult:
    completed_epoch: int
    global_optimizer_step: int
    best_epoch: int | None
    best_metric: float | None
    last_checkpoint: Path
    best_checkpoint: Path
    stopped_early: bool
    test_dataset_created: bool
    final_prototype_bank: PrototypeBank
    final_validation: TextFreeValidationResult
    train_epoch_results: tuple[TrainEpochResult, ...]


def _dataset_text_indices(dataset: ContextEEGDataset) -> set[int]:
    return {int(record.text_embedding_idx) for record in dataset.records}


def _dataset_group_ids(dataset: ContextEEGDataset) -> set[str]:
    return {
        dataset.split_index.assignment(
            dataset.outer_fold,
            int(record.text_embedding_idx),
        ).sentence_group_id
        for record in dataset.records
    }


def guard_fold_training_boundaries(
    train_dataset: ContextEEGDataset,
    validation_dataset: ContextEEGDataset,
) -> None:
    if train_dataset.role != "train":
        raise ValueError("Trainer requires role=train for its training Dataset")
    if validation_dataset.role != "validation":
        raise ValueError("Trainer requires role=validation; role=test is forbidden")
    if train_dataset.outer_fold != validation_dataset.outer_fold:
        raise ValueError("Training and validation outer folds differ")
    if train_dataset.include_context_targets is not True:
        raise ValueError("Training Dataset must include context targets")
    if validation_dataset.include_context_targets is not False:
        raise ValueError("Validation Dataset must not include context targets")
    if getattr(validation_dataset, "context_store_root", None) is not None:
        raise ValueError("Validation Dataset is wired to a forbidden text cache")
    train_text = _dataset_text_indices(train_dataset)
    validation_text = _dataset_text_indices(validation_dataset)
    intersection = train_text & validation_text
    if intersection:
        raise ValueError(
            f"Train/validation text_embedding_idx overlap: {sorted(intersection)[:5]}"
        )
    train_groups = _dataset_group_ids(train_dataset)
    validation_groups = _dataset_group_ids(validation_dataset)
    group_intersection = train_groups & validation_groups
    if group_intersection:
        raise ValueError(
            "Train/validation sentence_group_id overlap: "
            f"{sorted(group_intersection)[:5]}"
        )


class FoldTrainer:
    """One-fold trainer whose public data API has no outer-test input."""

    def __init__(
        self,
        *,
        train_dataset: ContextEEGDataset,
        validation_dataset: ContextEEGDataset,
        train_loader: Any,
        validation_loader: Any,
        eeg_encoder: nn.Module,
        text_projector: ContextTextProjector,
        aligner: ContextOTAligner,
        loss_module: ContextOTThreeLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: torch.amp.GradScaler,
        prototype_refresh: PrototypeRefreshCoordinator,
        validation_scorer: TextFreePrototypeScorer,
        eligibility: FoldKeywordEligibility,
        early_stopping: EarlyStopping,
        config: FoldTrainerConfig,
        resolved_config: Mapping[str, Any],
        asset_hashes: Mapping[str, str],
        subject_index_mapping: Mapping[str, int],
        keyword_index_mapping: Mapping[str, int],
        run_directory: str | Path,
        device: torch.device | str,
        dataloader_generator: torch.Generator | None = None,
        project_root: str | Path | None = None,
    ) -> None:
        config.validate()
        guard_fold_training_boundaries(train_dataset, validation_dataset)
        if getattr(train_loader, "dataset", train_dataset) is not train_dataset:
            raise ValueError("train_loader is not bound to train_dataset")
        if getattr(validation_loader, "dataset", validation_dataset) is not validation_dataset:
            raise ValueError("validation_loader is not bound to validation_dataset")
        if aligner.text_projector is not text_projector:
            raise ValueError("Aligner and prototype refresh must share one projector")
        if eligibility.outer_fold != train_dataset.outer_fold:
            raise ValueError("Eligibility outer fold disagrees with the datasets")
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.eeg_encoder = eeg_encoder
        self.text_projector = text_projector
        self.aligner = aligner
        self.loss_module = loss_module
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.prototype_refresh = prototype_refresh
        self.validation_scorer = validation_scorer
        self.eligibility = eligibility
        self.early_stopping = early_stopping
        self.config = config
        self.resolved_config = dict(resolved_config)
        self.asset_hashes = dict(asset_hashes)
        self.subject_index_mapping = dict(subject_index_mapping)
        self.keyword_index_mapping = dict(keyword_index_mapping)
        self.run_directory = Path(run_directory)
        self.device = torch.device(device)
        self.dataloader_generator = dataloader_generator
        self.project_root = project_root
        self.outer_fold = train_dataset.outer_fold
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self.checkpoint_directory = self.run_directory / "checkpoints"
        self.checkpoint_directory.mkdir(parents=True, exist_ok=True)
        self.logger = JsonlLogger(self.run_directory / "train_log.jsonl")
        self.validation_logger = JsonlLogger(
            self.run_directory / "validation_metrics.jsonl"
        )
        self.prototype_directory = self.run_directory / "prototype_banks"
        self.prototype_directory.mkdir(parents=True, exist_ok=True)
        self.eeg_encoder.to(self.device)
        self.aligner.to(self.device)

    def _expectations(self) -> CheckpointExpectations:
        return CheckpointExpectations(
            outer_fold=self.outer_fold,
            text_backend=self.text_projector.config.backend,
            input_channels=int(self.eeg_encoder.config.input_channels),
            subject_index_mapping=self.subject_index_mapping,
            keyword_index_mapping=self.keyword_index_mapping,
            asset_hashes=self.asset_hashes,
            config_sha256=canonical_sha256(self.resolved_config),
        )

    def _check_contributors(self, bank: PrototypeBank) -> None:
        contributors = {
            int(value)
            for value in bank.metadata["contributors"]["text_embedding_indices"]
        }
        train_text = _dataset_text_indices(self.train_dataset)
        validation_text = _dataset_text_indices(self.validation_dataset)
        if contributors != train_text:
            raise RuntimeError("Prototype contributors do not exactly match inner-train")
        if contributors & validation_text:
            raise RuntimeError("Prototype contributors overlap validation")

    def fit(
        self,
        *,
        resume_from: str | Path | None = None,
        stop_after_epoch: int | None = None,
    ) -> FoldTrainingResult:
        start_epoch = 0
        global_step = 0
        best_metrics: dict[str, float] = {}
        bank: PrototypeBank | None = None
        if resume_from is not None:
            loaded = load_training_checkpoint(
                resume_from,
                expected=self._expectations(),
                eeg_encoder=self.eeg_encoder,
                text_projector=self.text_projector,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                early_stopping=self.early_stopping,
                dataloader_generator=self.dataloader_generator,
                map_location="cpu",
            )
            start_epoch = loaded.epoch + 1
            global_step = loaded.global_optimizer_step
            best_metrics = loaded.best_validation_metrics
            bank = loaded.prototype_bank

        epoch_results: list[TrainEpochResult] = []
        final_validation: TextFreeValidationResult | None = None
        last_epoch = start_epoch - 1
        for epoch in range(start_epoch, self.config.epochs):
            if stop_after_epoch is not None and epoch > stop_after_epoch:
                break
            dataloader_epoch_start = (
                self.dataloader_generator.get_state()
                if self.dataloader_generator is not None
                else None
            )
            start_refresh = self.prototype_refresh.refresh(self.text_projector)
            bank = start_refresh.bank
            self._check_contributors(bank)
            self.logger.write(
                {
                    "event": "prototype_refresh",
                    "timing": "epoch_start",
                    "epoch": epoch,
                    "projector_hash": start_refresh.projector_hash,
                    "prototype_bank_hash": start_refresh.bank_hash,
                    "available_count": bank.available_count,
                }
            )
            train_result = train_one_epoch(
                epoch=epoch,
                global_optimizer_step=global_step,
                eeg_encoder=self.eeg_encoder,
                aligner=self.aligner,
                loss_module=self.loss_module,
                train_loader=self.train_loader,
                prototype_bank=bank,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                device=self.device,
                config=self.config.train_epoch,
                logger=self.logger,
            )
            epoch_results.append(train_result)
            global_step = train_result.global_optimizer_step

            validation_refresh = self.prototype_refresh.refresh(self.text_projector)
            bank = validation_refresh.bank
            self._check_contributors(bank)
            self.prototype_refresh.assert_current(bank, self.text_projector)
            self.logger.write(
                {
                    "event": "prototype_refresh",
                    "timing": "after_train_before_validation",
                    "epoch": epoch,
                    "projector_hash": validation_refresh.projector_hash,
                    "prototype_bank_hash": validation_refresh.bank_hash,
                    "available_count": bank.available_count,
                }
            )
            final_validation = run_text_free_validation(
                eeg_encoder=self.eeg_encoder,
                validation_loader=self.validation_loader,
                prototype_bank=bank,
                scorer=self.validation_scorer,
                eligibility=self.eligibility,
                device=self.device,
                max_batches=self.config.max_validation_batches,
            )
            improved = self.early_stopping.update(
                final_validation.metrics,
                epoch=epoch,
            )
            if improved:
                best_metrics = dict(final_validation.metrics)
            core_group_report = final_validation.reports["core/context_group"]
            validation_event = {
                "event": "validation_epoch",
                "epoch": epoch,
                "scorer_temperature": final_validation.scorer_temperature,
                "view_count": final_validation.tables["view"].scores.shape[0],
                "sentence_occurrence_count": final_validation.tables[
                    "sentence_occurrence"
                ].scores.shape[0],
                "context_group_count": final_validation.tables[
                    "context_group"
                ].scores.shape[0],
                "core_context_group_macro_auprc": core_group_report.macro_auprc,
                "valid_core_keyword_count": core_group_report.valid_keyword_count,
                "total_core_keyword_count": core_group_report.total_tier_keyword_count,
                "core_positive_counts": core_group_report.positive_counts.tolist(),
                "core_negative_counts": core_group_report.negative_counts.tolist(),
                "core_validity_reasons": list(core_group_report.validity_reasons),
                "main_valid_keyword_count": final_validation.reports[
                    "main/context_group"
                ].valid_keyword_count,
                "extended_valid_keyword_count": final_validation.reports[
                    "extended/context_group"
                ].valid_keyword_count,
                "prototype_bank_hash": validation_refresh.bank_hash,
                "projector_hash": validation_refresh.projector_hash,
                "best_checkpoint_updated": improved,
                "early_stopping_bad_epochs": self.early_stopping.bad_epochs,
            }
            self.logger.write(validation_event)
            self.validation_logger.write(validation_event)
            save_prototype_bank(
                bank,
                self.prototype_directory / f"epoch_{epoch:04d}",
            )
            checkpoint_arguments = {
                "outer_fold": self.outer_fold,
                "epoch": epoch,
                "global_optimizer_step": global_step,
                "eeg_encoder": self.eeg_encoder,
                "text_projector": self.text_projector,
                "optimizer": self.optimizer,
                "scheduler": self.scheduler,
                "scaler": self.scaler,
                "prototype_bank": bank,
                "resolved_config": self.resolved_config,
                "asset_hashes": self.asset_hashes,
                "subject_index_mapping": self.subject_index_mapping,
                "keyword_index_mapping": self.keyword_index_mapping,
                "current_validation_metrics": final_validation.metrics,
                "best_validation_metrics": best_metrics,
                "early_stopping": self.early_stopping,
                "dataloader_generator": self.dataloader_generator,
                "dataloader_epoch_start_state": dataloader_epoch_start,
                "project_root": self.project_root,
            }
            last_path = self.checkpoint_directory / "last.pt"
            save_training_checkpoint(last_path, **checkpoint_arguments)
            best_path = self.checkpoint_directory / "best.pt"
            if improved:
                # Re-serialize instead of symlinking so Windows artifacts are portable.
                shutil.copy2(last_path, best_path)
            last_epoch = epoch
            if self.early_stopping.should_stop:
                break
        if bank is None or final_validation is None:
            raise RuntimeError("No epoch was run; checkpoint already reached configured epochs")
        return FoldTrainingResult(
            completed_epoch=last_epoch,
            global_optimizer_step=global_step,
            best_epoch=self.early_stopping.best_epoch,
            best_metric=self.early_stopping.best_value,
            last_checkpoint=self.checkpoint_directory / "last.pt",
            best_checkpoint=self.checkpoint_directory / "best.pt",
            stopped_early=self.early_stopping.should_stop,
            test_dataset_created=False,
            final_prototype_bank=bank,
            final_validation=final_validation,
            train_epoch_results=tuple(epoch_results),
        )
