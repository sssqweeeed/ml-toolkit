"""Selection, optimization and high-level workflow settings."""

from dataclasses import dataclass, field
from math import isfinite
from typing import Literal

from .._console import Verbosity
from .reporting import ReportingConfig
from .tracking import TrackingConfig


@dataclass(frozen=True, kw_only=True, slots=True)
class TuningConfig:
    """Configure each split's independent Optuna study.

    Parameters
    ----------
    metric : str
        Metric of the current split's validation rows, selected by its score callback.
    direction : {'minimize', 'maximize'}
        Direction of the objective.
    n_trials : int, default 50
        Positive maximum number of trials per split.
    timeout_seconds : float or None, default None
        Positive time limit per split; a running trial may exceed it.
    seed : int or None, default None
        Nonnegative TPE sampler seed, independent of model seeds.
    """

    metric: str
    direction: Literal["minimize", "maximize"]
    n_trials: int = 50
    timeout_seconds: float | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.n_trials < 1:
            raise ValueError("n_trials must be positive")
        if not self.metric.strip():
            raise ValueError("metric must not be empty")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if self.timeout_seconds is not None and (
            not isfinite(self.timeout_seconds) or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive")


@dataclass(frozen=True, kw_only=True, slots=True)
class TrainingConfig:
    """Configure parameter tuning, additional reports and model packaging.

    Parameters
    ----------
    tuning : TuningConfig or None, default None
        Independent search per validation split, requiring param_space.
    reporting : ReportingConfig
        Additional metric and artifact settings.
    tracking : TrackingConfig
        Model packaging settings.
    fit_on_test : bool, default False
        After independent test evaluation, fit an additional fresh model on
        all eligible data including test. Requires a held-out test sample.
    verbosity : {'silent', 'normal', 'verbose', 'debug'}, default 'normal'
        Console output for this training run: none, milestones and summary,
        split/trial details, or internal diagnostics. Writes to stderr without
        configuring Python logging or changing CatBoost, Optuna or MLflow output.
        Warnings, exceptions and application logging remain independent.

    Notes
    -----
    Both post-validation fits use the last split's features and refit parameters.
    """

    tuning: TuningConfig | None = None
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    fit_on_test: bool = False
    verbosity: Verbosity = "normal"

    def __post_init__(self) -> None:
        if self.verbosity not in ("silent", "normal", "verbose", "debug"):
            raise ValueError("verbosity must be one of: silent, normal, verbose, debug")
