from __future__ import annotations

from dataclasses import asdict, dataclass


PRIMARY_VALIDATION_METRIC = "validation/core/context_group/macro_auprc"


@dataclass
class EarlyStopping:
    patience: int = 10
    min_delta: float = 0.001
    metric: str = PRIMARY_VALIDATION_METRIC
    mode: str = "max"
    best_value: float | None = None
    best_epoch: int | None = None
    bad_epochs: int = 0

    def __post_init__(self) -> None:
        if self.metric != PRIMARY_VALIDATION_METRIC:
            raise ValueError(
                f"Only the preregistered metric is allowed: {PRIMARY_VALIDATION_METRIC}"
            )
        if self.mode != "max":
            raise ValueError("Core context-group macro-AUPRC must be maximized")
        if self.patience < 0 or self.min_delta < 0:
            raise ValueError("patience and min_delta must be non-negative")

    def update(self, metrics: dict[str, float], *, epoch: int) -> bool:
        if self.metric not in metrics:
            raise KeyError(f"Missing early-stopping metric: {self.metric}")
        value = float(metrics[self.metric])
        if value != value:
            raise ValueError("Early-stopping metric is NaN")
        # Strict min_delta means equal scores retain the earlier epoch.
        improved = self.best_value is None or value > self.best_value + self.min_delta
        if improved:
            self.best_value = value
            self.best_epoch = epoch
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience if self.bad_epochs else False

    def state_dict(self) -> dict[str, object]:
        return asdict(self)

    def load_state_dict(self, state: dict[str, object]) -> None:
        restored = EarlyStopping(**state)
        self.__dict__.update(restored.__dict__)
