"""Selection, optimization and high-level workflow settings."""

from dataclasses import dataclass, field
from datetime import date, datetime
from math import isfinite
from typing import Literal

from .reporting import ReportingConfig
from .splits import HoldoutConfig, TimeSeriesCVConfig
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
    """Configure selection, validation, tuning, reporting and final fitting.

    Parameters
    ----------
    validation : HoldoutConfig or TimeSeriesCVConfig
        Required strategy defining independent selection and tuning runs.
    test_size : int, float, str, datetime.date, or None, default None
        Trailing test row count or fraction of the original data, rounded up.
        A date or ``YYYY-MM-DD`` string starts test on that calendar day,
        inclusive, in the timezone of the data's timestamps. Strings are
        normalized to dates during construction; datetimes are not accepted.
        Both training and test must retain at least one row.
        None uses all rows for training and performs no test evaluation.
    tuning : TuningConfig or None, default None
        Independent search per split, requiring a separately supplied param_space.
    reporting : ReportingConfig
        Diagnostics and custom evaluation settings; defaults to ReportingConfig.
    tracking : TrackingConfig
        Model packaging settings; defaults to TrackingConfig.

    Notes
    -----
    Data and executing dependencies remain explicit function arguments.
    Final fitting uses the last validation split's features, iteration budget
    and optional threshold, on all rows except the test tail.

    Examples
    --------
    >>> config = TrainingConfig(validation=HoldoutConfig(), test_size=0.2)
    >>> config.tuning is None
    True
    >>> config = TrainingConfig(validation=HoldoutConfig(), test_size="2025-01-01")
    >>> config.test_size
    datetime.date(2025, 1, 1)
    """

    validation: HoldoutConfig | TimeSeriesCVConfig
    test_size: int | float | str | date | None = None
    tuning: TuningConfig | None = None
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    def __post_init__(self) -> None:
        if isinstance(self.test_size, str):
            try:
                start_date = date.fromisoformat(self.test_size)
            except ValueError as error:
                raise ValueError("test_size date must use YYYY-MM-DD format") from error
            if start_date.isoformat() != self.test_size:
                raise ValueError("test_size date must use YYYY-MM-DD format")
            object.__setattr__(self, "test_size", start_date)
        elif isinstance(self.test_size, datetime):
            raise ValueError("test_size must be a calendar date without a time")
        elif isinstance(self.test_size, (int, float)):
            HoldoutConfig(val_size=self.test_size)
        selected = self.reporting.validation_splits
        if isinstance(selected, tuple):
            first, last = 1, 1
            if isinstance(self.validation, TimeSeriesCVConfig):
                first = self.validation.skip_splits + 1
                last = self.validation.n_splits
            if any(not first <= index <= last for index in selected):
                raise ValueError(
                    "Diagnostic splits must be evaluated validation split indices"
                )
