"""Selection, optimization and high-level workflow settings."""

from dataclasses import dataclass, field
from datetime import date, datetime
from math import isfinite
from typing import Literal

from .reporting import ReportingConfig
from .splits import HoldoutConfig, TimeSeriesCVConfig
from .tracking import TrackingConfig


@dataclass(frozen=True, kw_only=True, slots=True)
class RFEFeatureSelectionConfig:
    """Search feature subsets on the supplied train/validation split.

    Parameters
    ----------
    min_features : int
        Positive lower bound on visited subset sizes, not a required result size.
    metric : str
        Key returned by the metric callable for comparing subsets.
    direction : {'minimize', 'maximize'}
        Direction of the selection objective.
    max_fits : int, default 6
        Maximum fits, including the complete feature set.
    score_tolerance : float, default 0.0
        Nonnegative absolute distance from the best score. Choose the smallest
        visited subset within this distance.
    protected_features : tuple[str, ...], default ()
        Unique columns retained in every subset, at most min_features names.

    Notes
    -----
    Each split selects features once before its own Optuna study. Final fitting
    reuses the last split's selected columns.
    """

    min_features: int
    metric: str
    direction: Literal["minimize", "maximize"]
    max_fits: int = 6
    score_tolerance: float = 0.0
    protected_features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.min_features < 1 or self.max_fits < 1:
            raise ValueError("min_features and max_fits must be positive")
        if not self.metric.strip():
            raise ValueError("metric must not be empty")
        if any(not name.strip() for name in self.protected_features) or len(
            set(self.protected_features)
        ) != len(self.protected_features):
            raise ValueError("protected_features must contain unique nonempty names")
        if not isfinite(self.score_tolerance) or self.score_tolerance < 0:
            raise ValueError("score_tolerance must be nonnegative")
        if len(self.protected_features) > self.min_features:
            raise ValueError("protected_features must not exceed min_features")


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
    feature_selection : RFEFeatureSelectionConfig or None, default None
        Selection once per train/validation split; None keeps all columns.
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
    >>> config.feature_selection is None
    True
    >>> config = TrainingConfig(validation=HoldoutConfig(), test_size="2025-01-01")
    >>> config.test_size
    datetime.date(2025, 1, 1)
    """

    validation: HoldoutConfig | TimeSeriesCVConfig
    test_size: int | float | str | date | None = None
    feature_selection: RFEFeatureSelectionConfig | None = None
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
