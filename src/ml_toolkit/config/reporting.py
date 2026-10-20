"""Task-specific diagnostics and custom post-fit reporting policies."""

from dataclasses import dataclass, field
from math import isfinite
from typing import Literal

from .._time import PERIOD_INTERVALS
from ..feature_levels import FeatureLevelSpec
from ..types import TimePeriod


@dataclass(frozen=True, kw_only=True, slots=True)
class FigureConfig:
    """Configure figure dimensions and file output.

    Parameters
    ----------
    figsize : tuple[float, float] or None, default None
        Positive finite width and height in inches. None preserves each plot's
        native or adaptive dimensions.
    dpi : int, default 150
        Positive export resolution in dots per inch.
    format : {'png', 'svg', 'pdf'}, default 'png'
        Output format. Explicit figure settings replace the output path suffix
        to match this format. All figures are closed after saving.
    """

    figsize: tuple[float, float] | None = None
    dpi: int = 150
    format: Literal["png", "svg", "pdf"] = "png"

    def __post_init__(self) -> None:
        if self.dpi < 1:
            raise ValueError("dpi must be positive")
        if self.figsize is not None and any(
            not isfinite(value) or value <= 0 for value in self.figsize
        ):
            raise ValueError("figsize dimensions must be positive")


@dataclass(frozen=True, kw_only=True, slots=True)
class NumericalFeatureDistributionConfig:
    """Choose time-based numerical-feature distributions and summary curves.

    Parameters
    ----------
    features : tuple[str, ...] or None, default None
        None selects all numerical features; an empty tuple selects none.
        Explicit names must exist and be numerical in both splits.
    quantile_bands : tuple[tuple[float, float], ...]
        Ordered unique pairs with 0 <= lower < upper <= 1. Defaults to 5-95%
        and 25-75%. An empty tuple disables shaded bands.
    show_mean : bool, default True
        Display the unweighted time-based mean.
    show_median : bool, default True
        Display the unweighted time-based median.

    Notes
    -----
    Null and non-finite values are omitted. Calendar-period summaries and curves
    do not use exposure weights.
    """

    features: tuple[str, ...] | None = None
    quantile_bands: tuple[tuple[float, float], ...] = ((0.05, 0.95), (0.25, 0.75))
    show_mean: bool = True
    show_median: bool = True

    def __post_init__(self) -> None:
        if any(not 0 <= lo < hi <= 1 for lo, hi in self.quantile_bands):
            raise ValueError("quantile_bands require 0 <= lower < upper <= 1")
        if len(set(self.quantile_bands)) != len(self.quantile_bands):
            raise ValueError("quantile_bands must be unique")

        if self.features is not None:
            names = self.features
            if any(not name.strip() for name in names) or len(set(names)) != len(names):
                raise ValueError("features must contain unique nonempty names")


@dataclass(frozen=True, kw_only=True, slots=True)
class ShapConfig:
    """Configure SHAP sampling and displayed feature contributions.

    Parameters
    ----------
    max_samples : int, default 10000
        Positive maximum row count sampled without replacement from validation or test.
    seed : int, default 0
        Nonnegative sampling seed, independent of model and tuning seeds.
    max_display : int, default 10
        Positive feature limit for importance and beeswarm plots.
    dependence_features : tuple[str, ...], default ()
        Unique feature names for dependence plots. Empty disables them.

    Notes
    -----
    Requires the shap extra and a supported single-output native tree model.
    Explanations use the native raw-output scale.
    """

    max_samples: int = 10_000
    seed: int = 0
    max_display: int = 10
    dependence_features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.max_samples < 1:
            raise ValueError("max_samples must be positive")
        if self.max_display < 1:
            raise ValueError("max_display must be positive")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        names = self.dependence_features
        if any(not name.strip() for name in names) or len(set(names)) != len(names):
            raise ValueError("dependence_features must contain unique nonempty names")


@dataclass(frozen=True, kw_only=True, slots=True)
class CommonPlotsConfig:
    """Configure shared feature diagnostics and figure exports for any task.

    Parameters
    ----------
    features : tuple[FeatureLevelSpec, ...] or None, default None
        Shared feature definitions for PSI and observed/fitted. None selects all
        input features using categorical declarations and dtype; an empty tuple
        disables both feature diagnostics. Explicit names must exist after
        feature selection. Order is preserved.
    psi : bool, default True
        Calculate PSI tables and plot scores and time-based level distributions.
    observed_fitted : bool, default True
        Plot weighted observed/fitted means for train and validation/test on the
        same reference-fitted levels. Available for every task; the caller is
        responsible for the interpretation of targets, including PU labels.
    period : TimePeriod, default "month"
        Calendar grouping for all temporal diagnostics in the input timezone.
        Minute/hour require Datetime values; weeks start on Monday.
        Reference timestamps must strictly precede current timestamps; a split
        inside one grouping period is allowed.
    n_bins : int, default 10
        Positive requested number of reference quantile bins for both diagnostics.
    epsilon : float, default 1e-6
        PSI share floor strictly between zero and one.
    figure : FigureConfig
        Shared export settings; defaults to a fresh FigureConfig.
    numerical_distributions : NumericalFeatureDistributionConfig or None
        Independently configured raw numerical distributions; defaults to all
        numerical features. None disables these plots.
    shap : ShapConfig or None, default None
        Optional SHAP diagnostics; None disables SHAP calculation and plots.

    Notes
    -----
    Levels are fitted once on training rows and reused by both diagnostics.
    Turning off PSI does not disable observed/fitted or vice versa. Setting the
    enclosing diagnostics' plots to None disables all plots and PSI tables.

    Examples
    --------
    >>> common = CommonPlotsConfig(
    ...     features=(FeatureLevelSpec("age", "numerical", lower_bound=0),),
    ...     psi=False,
    ...     observed_fitted=True,
    ... )
    >>> common.n_bins
    10
    """

    features: tuple[FeatureLevelSpec, ...] | None = None
    psi: bool = True
    observed_fitted: bool = True
    period: TimePeriod = "month"
    n_bins: int = 10
    epsilon: float = 1e-6
    figure: FigureConfig = field(default_factory=FigureConfig)
    numerical_distributions: NumericalFeatureDistributionConfig | None = field(
        default_factory=NumericalFeatureDistributionConfig
    )
    shap: ShapConfig | None = None

    def __post_init__(self) -> None:
        if self.period not in PERIOD_INTERVALS:
            raise ValueError("Unsupported time period")
        if self.n_bins < 1:
            raise ValueError("n_bins must be positive")
        if not 0 < self.epsilon < 1:
            raise ValueError("epsilon must be between 0 and 1")
        if self.features is not None:
            names = tuple(feature.name for feature in self.features)
            if len(set(names)) != len(names):
                raise ValueError("features must contain unique names")


@dataclass(frozen=True, kw_only=True, slots=True)
class ClassificationPlotsConfig:
    """Choose supervised binary-classification figures.

    Parameters
    ----------
    common : CommonPlotsConfig
        Shared figures and export settings; defaults to CommonPlotsConfig.
    precision_recall : bool, default True
        Display the weighted precision-recall curve.
    f1_threshold : bool, default True
        Display weighted F1 across distinct score thresholds.
    calibration : bool, default True
        Display weighted observed rates against predicted probabilities.
    gains : bool, default True
        Display exposure-weighted cumulative gains for nonnegative outcomes.
    lift : bool, default True
        Display exposure-weighted cumulative lift.
    time_observed_fitted : bool, default True
        Display time-based exposure-weighted observed and fitted means.
    max_curve_points : int, default 1000
        Positive rendering limit for PR and F1; calculations retain all points.
    calibration_bins : int, default 10
        Positive requested number of probability-quantile calibration bins.
    histogram_bins : int, default 30
        Positive histogram-bin count in the calibration figure.
    """

    common: CommonPlotsConfig = field(default_factory=CommonPlotsConfig)
    precision_recall: bool = True
    f1_threshold: bool = True
    calibration: bool = True
    gains: bool = True
    lift: bool = True
    time_observed_fitted: bool = True
    max_curve_points: int = 1_000
    calibration_bins: int = 10
    histogram_bins: int = 30

    def __post_init__(self) -> None:
        if self.max_curve_points < 1:
            raise ValueError("max_curve_points must be positive")
        if self.calibration_bins < 1:
            raise ValueError("calibration_bins must be positive")
        if self.histogram_bins < 1:
            raise ValueError("histogram_bins must be positive")


@dataclass(frozen=True, kw_only=True, slots=True)
class RegressionPlotsConfig:
    """Choose regression, frequency, or severity figures.

    Parameters
    ----------
    common : CommonPlotsConfig
        Shared figures and export settings; defaults to CommonPlotsConfig.
    gains : bool, default False
        Enable exposure-weighted gains. Requires nonnegative validation targets
        with positive weighted total; invalid inputs raise rather than skip.
    lift : bool, default False
        Enable cumulative lift with the same target constraints as gains.
    time_observed_fitted : bool, default True
        Display time-based exposure-weighted observed and fitted means.

    Notes
    -----
    Ranking plots default off so ordinary signed regression targets are valid.
    Frequency and severity recipes can enable them explicitly.
    """

    common: CommonPlotsConfig = field(default_factory=CommonPlotsConfig)
    gains: bool = False
    lift: bool = False
    time_observed_fitted: bool = True


@dataclass(frozen=True, kw_only=True, slots=True)
class PUPlotsConfig:
    """Choose positive-unlabeled figures without supervised assumptions.

    Parameters
    ----------
    common : CommonPlotsConfig
        Shared figures and export settings; defaults to CommonPlotsConfig.
    score_distribution : bool, default True
        Display separately normalized weighted positive and unlabeled scores.
    positive_capture : bool, default True
        Display known-positive capture versus unlabeled review capacity.
    histogram_bins : int, default 30
        Positive number of equal-width score bins in [0, 1].

    Notes
    -----
    No supervised gains, lift, calibration, or F1 fields are accepted.
    """

    common: CommonPlotsConfig = field(default_factory=CommonPlotsConfig)
    score_distribution: bool = True
    positive_capture: bool = True
    histogram_bins: int = 30

    def __post_init__(self) -> None:
        if self.histogram_bins < 1:
            raise ValueError("histogram_bins must be positive")


@dataclass(frozen=True, kw_only=True, slots=True)
class ClassificationDiagnosticsConfig:
    """Configure binary-classification diagnostics.

    Parameters
    ----------
    plots : ClassificationPlotsConfig or None
        Defaults to ClassificationPlotsConfig. Shared features, PSI and
        observed/fitted are configured in plots.common. None disables every
        figure and PSI table.
    """

    plots: ClassificationPlotsConfig | None = field(
        default_factory=ClassificationPlotsConfig
    )

    @property
    def task(self) -> Literal["classification"]:
        """Return the fixed supervised classification task identifier."""
        return "classification"


@dataclass(frozen=True, kw_only=True, slots=True)
class RegressionDiagnosticsConfig:
    """Configure regression-family diagnostics.

    Parameters
    ----------
    task : {'regression', 'frequency', 'severity'}, default 'regression'
        Task label for diagnostic metadata.
    plots : RegressionPlotsConfig or None
        Defaults to RegressionPlotsConfig. Shared features, PSI and
        observed/fitted are configured in plots.common. None disables every
        figure and PSI table.
    """

    task: Literal["regression", "frequency", "severity"] = "regression"
    plots: RegressionPlotsConfig | None = field(default_factory=RegressionPlotsConfig)


@dataclass(frozen=True, kw_only=True, slots=True)
class PUDiagnosticsConfig:
    """Configure positive-unlabeled diagnostics.

    Parameters
    ----------
    plots : PUPlotsConfig or None
        Defaults to PUPlotsConfig. Shared features, PSI and observed/fitted are
        configured in plots.common, without task-specific restrictions. The
        caller decides whether comparing PU labels to predictions is meaningful.
        None disables every figure and PSI table.
    """

    plots: PUPlotsConfig | None = field(default_factory=PUPlotsConfig)

    @property
    def task(self) -> Literal["pu_classification"]:
        """Return the fixed positive-unlabeled classification identifier."""
        return "pu_classification"


DiagnosticsConfig = (
    ClassificationDiagnosticsConfig | RegressionDiagnosticsConfig | PUDiagnosticsConfig
)


@dataclass(frozen=True, kw_only=True, slots=True)
class PostFitEvaluationConfig:
    """Control calls to separately supplied custom post-fit evaluators.

    Parameters
    ----------
    stages : tuple[{'tuning', 'validation', 'final'}, ...], default ('validation', 'final')
        Unique enabled stages. An empty tuple disables all calls.
    splits : tuple[{'train', 'val', 'test'}, ...], default ('train', 'val', 'test')
        Enabled splits. Final fitting evaluates train and optional test. Empty
        disables all calls; unavailable splits are skipped.
    write_artifacts : bool, default True
        Call the evaluator's optional artifact writer after scalar evaluation.
        False disables all artifact files while retaining enabled scalar metrics.

    Notes
    -----
    The policy applies to every supplied evaluator, in list order. Instances
    are not copied or reset. Custom scores are logged separately by evaluator
    name and do not become validation averages or optimization objectives.
    """

    stages: tuple[Literal["tuning", "validation", "final"], ...] = (
        "validation",
        "final",
    )
    splits: tuple[Literal["train", "val", "test"], ...] = ("train", "val", "test")
    write_artifacts: bool = True

    def __post_init__(self) -> None:
        if len(set(self.stages)) != len(self.stages) or len(set(self.splits)) != len(
            self.splits
        ):
            raise ValueError("post-fit stages and splits must be unique")


@dataclass(frozen=True, kw_only=True, slots=True)
class ReportingConfig:
    """Choose validation/test diagnostics and custom evaluation.

    Parameters
    ----------
    diagnostics : DiagnosticsConfig or None, default None
        Optional task-specific PSI and figures. The same reports apply to test.
    validation_splits : {'all', 'last'} or tuple[int, ...], default 'last'
        Validation splits to diagnose. Explicit indices are positive, unique,
        one-based original indices before skipping. Does not restrict test or
        custom evaluation. Built-in diagnostics do not run during tuning.
        Split availability is checked only when diagnostics and their plots
        are enabled.
    post_fit : PostFitEvaluationConfig or None, default None
        Custom evaluator stage, split and artifact policy. None uses the standard
        policy only when an evaluator is supplied: validation and final stages,
        including test when available. Without an evaluator no calls run.
        Empty stages or splits explicitly disable calls.
    """

    diagnostics: DiagnosticsConfig | None = None
    validation_splits: Literal["all", "last"] | tuple[int, ...] = "last"
    post_fit: PostFitEvaluationConfig | None = None

    def __post_init__(self) -> None:
        if isinstance(self.validation_splits, tuple) and (
            not self.validation_splits
            or any(index < 1 for index in self.validation_splits)
            or len(set(self.validation_splits)) != len(self.validation_splits)
        ):
            raise ValueError(
                "validation_splits must contain unique positive one-based indices"
            )
