"""Typed, immutable settings shared by standalone APIs and training workflows."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from math import isfinite
from pathlib import Path
from types import UnionType
from typing import Literal, Union, get_args, get_origin, get_type_hints

from .psi.config import PSIFeature


def _matches(value, annotation):
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (Union, UnionType):
        return any(_matches(value, arg) for arg in args)
    if origin is Literal:
        return any(type(value) is type(arg) and value == arg for arg in args)
    if origin is tuple:
        if not isinstance(value, tuple):
            return False
        if len(args) == 2 and args[1] is Ellipsis:
            return all(_matches(item, args[0]) for item in value)
        return len(value) == len(args) and all(
            _matches(v, a) for v, a in zip(value, args)
        )
    if annotation is float:
        return type(value) in (float, int) and isfinite(value)
    if annotation in (int, bool, str):
        return type(value) is annotation
    return isinstance(value, annotation)


class _Config:
    __slots__ = ()

    def __post_init__(self):
        hints = get_type_hints(type(self))
        for item in fields(self):
            value = getattr(self, item.name)
            if not _matches(value, hints[item.name]):
                raise TypeError(
                    f"{type(self).__name__}.{item.name} has an invalid type or value"
                )
            if (
                item.name
                in {
                    "min_features",
                    "max_fits",
                    "n_splits",
                    "test_size",
                    "max_train_size",
                    "n_trials",
                    "n_bins",
                    "dpi",
                    "max_curve_points",
                    "calibration_bins",
                    "histogram_bins",
                    "max_samples",
                    "max_display",
                    "input_example_rows",
                }
                and value is not None
                and value < 1
            ):
                raise ValueError(f"{item.name} must be positive")
            if (
                item.name in {"skip_splits", "gap", "seed"}
                and value is not None
                and value < 0
            ):
                raise ValueError(f"{item.name} must be nonnegative")
            if (
                item.name in {
                    "metric", "metric_name", "model_name", "registered_model_name"
                }
                and value is not None
                and not value.strip()
            ):
                raise ValueError(f"{item.name} must not be empty")
            if item.name == "features" and value is not None:
                names = tuple(v.name if isinstance(v, PSIFeature) else v for v in value)
                _names(names, "features")
            if (
                item.name in {
                    "dependence_features", "psi_distribution_features",
                    "protected_features",
                }
                and value is not None
            ):
                _names(value, item.name)


def _names(names, name):
    if any(not value.strip() for value in names) or len(names) != len(set(names)):
        raise ValueError(f"{name} must contain unique, nonempty names")


@dataclass(frozen=True, kw_only=True, slots=True)
class RFEFeatureSelectionConfig(_Config):
    """Configure adapter-driven batch RFE on an internal chronological holdout.

    Parameters
    ----------
    min_features : int
        Positive lower bound on visited feature counts, at most the input count.
        The selected subset may be larger, including all input features.
    metric_name : str
        Nonempty key returned by the existing training metric callable.
    direction : {'minimize', 'maximize'}
        Direction used to compare that metric across feature subsets.
    max_fits : int, default 6
        Positive maximum number of adapter fits, including the full feature set.
        One evaluates only the full set. Rounded geometric sizes may use fewer.
    validation_fraction : float, default 0.25
        Trailing fraction of selection rows reserved for internal validation,
        strictly between zero and one. Outer CV validation rows are excluded.
    gap : int, default 0
        Nonnegative rows omitted immediately before internal validation.
    score_tolerance : float, default 0.0
        Nonnegative absolute distance from the best score. Select the smallest
        visited subset within this distance, resolving ties by visit order.
    protected_features : tuple[str, ...], default ()
        Unique nonempty input column names that must remain in every subset.
        Their count must not exceed min_features.

    Notes
    -----
    Selection runs once on the first evaluated fold's training rows; the result
    is reused by CV, tuning and final fitting. Parameters and split rows remain
    fixed. The feature ranker is an executing dependency supplied separately.
    """

    min_features: int
    metric_name: str
    direction: Literal["minimize", "maximize"]
    max_fits: int = 6
    validation_fraction: float = 0.25
    gap: int = 0
    score_tolerance: float = 0.0
    protected_features: tuple[str, ...] = ()

    def __post_init__(self):
        _Config.__post_init__(self)
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.score_tolerance < 0:
            raise ValueError("score_tolerance must be nonnegative")
        if len(self.protected_features) > self.min_features:
            raise ValueError("protected_features count must not exceed min_features")


@dataclass(frozen=True, kw_only=True, slots=True)
class CrossValidationConfig(_Config):
    """Configure expanding chronological cross-validation.

    Parameters
    ----------
    n_splits : int
        Number of generated folds, at least two.
    skip_splits : int, default 0
        Initial folds omitted from evaluation; smaller than n_splits.
    gap : int, default 0
        Number of rows excluded between each training and validation partition.
    test_size : int or None, default None
        Positive validation row count per fold; None uses TimeSeriesSplit's
        default. This is not an independent final test sample.
    max_train_size : int or None, default None
        Positive maximum training row count; None uses all preceding rows.

    Notes
    -----
    Sizes are row counts, not calendar durations. Input timestamps must be
    non-null and ascending. Fold indices are one-based before skipping.
    """

    n_splits: int
    skip_splits: int = 0
    gap: int = 0
    test_size: int | None = None
    max_train_size: int | None = None

    def __post_init__(self):
        _Config.__post_init__(self)
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if self.skip_splits >= self.n_splits:
            raise ValueError("skip_splits must be smaller than n_splits")


@dataclass(frozen=True, kw_only=True, slots=True)
class TuningConfig(_Config):
    """Configure Optuna's objective and sequential search budget.

    Parameters
    ----------
    metric : str
        Nonempty key returned by the supplied metric in CV mean_metrics.
    direction : {'minimize', 'maximize'}
        Optimization direction; required to avoid an implicit objective choice.
    n_trials : int, default 50
        Positive maximum number of trials.
    timeout_seconds : float or None, default None
        Positive wall-clock limit in seconds. An active trial may finish after
        the limit. None imposes no time limit.
    seed : int or None, default None
        Nonnegative seed for the default TPE sampler. None leaves it unseeded.
        Does not set the model's native random seed.
    """

    metric: str
    direction: Literal["minimize", "maximize"]
    n_trials: int = 50
    timeout_seconds: float | None = None
    seed: int | None = None

    def __post_init__(self):
        _Config.__post_init__(self)
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


@dataclass(frozen=True, kw_only=True, slots=True)
class FinalFitConfig(_Config):
    """Choose the iteration budget for fitting all retained rows.

    Parameters
    ----------
    iteration_policy : {'cv_median', 'provided'}, default 'cv_median'
        cv_median uses the median available zero-based best iteration plus one,
        replacing an existing iterations, n_estimators, or num_trees parameter.
        If no index or iteration key exists, the supplied budget is retained.
        provided always retains the supplied iteration budget.

    Notes
    -----
    Final fitting has no validation split: early stopping is removed and
    use_best_model is disabled under both policies.
    """

    iteration_policy: Literal["cv_median", "provided"] = "cv_median"


@dataclass(frozen=True, kw_only=True, slots=True)
class PSIOptions(_Config):
    """Configure diagnostic PSI with optional explicit feature definitions.

    Parameters
    ----------
    n_bins : int, default 10
        Positive requested reference quantile-bin count. Ties may reduce it.
    epsilon : float, default 1e-6
        Share floor strictly between zero and one for logarithmic PSI scoring.
    features : tuple[PSIFeature, ...] or None, default None
        None infers every input feature from categorical declarations and dtype.
        An explicit nonempty tuple selects features and supplies their bounds
        and sentinel rules. Numerical boundaries are fitted on training only.
    """

    n_bins: int = 10
    epsilon: float = 1e-6
    features: tuple[PSIFeature, ...] | None = None

    def __post_init__(self):
        _Config.__post_init__(self)
        if not 0 < self.epsilon < 1:
            raise ValueError("epsilon must be between 0 and 1")
        if self.features == ():
            raise ValueError(
                "PSI features must not be empty; use psi=None to disable PSI"
            )


@dataclass(frozen=True, kw_only=True, slots=True)
class FigureConfig(_Config):
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

    def __post_init__(self):
        _Config.__post_init__(self)
        if self.figsize is not None and any(value <= 0 for value in self.figsize):
            raise ValueError("figsize dimensions must be positive")


@dataclass(frozen=True, kw_only=True, slots=True)
class NumericalFeatureDistributionConfig(_Config):
    """Choose monthly numerical-feature distributions and summary curves.

    Parameters
    ----------
    features : tuple[str, ...] or None, default None
        None selects all numerical features; an empty tuple selects none.
        Explicit names must exist and be numerical in both splits.
    quantile_bands : tuple[tuple[float, float], ...]
        Ordered unique pairs with 0 <= lower < upper <= 1. Defaults to 5-95%
        and 25-75%. An empty tuple disables shaded bands.
    show_mean : bool, default True
        Display the unweighted monthly mean.
    show_median : bool, default True
        Display the unweighted monthly median.

    Notes
    -----
    Null and non-finite values are omitted. Calendar-month summaries and curves
    do not use exposure weights.
    """

    features: tuple[str, ...] | None = None
    quantile_bands: tuple[tuple[float, float], ...] = ((0.05, 0.95), (0.25, 0.75))
    show_mean: bool = True
    show_median: bool = True

    def __post_init__(self):
        _Config.__post_init__(self)
        if any(not 0 <= lo < hi <= 1 for lo, hi in self.quantile_bands):
            raise ValueError("quantile_bands require 0 <= lower < upper <= 1")
        if len(set(self.quantile_bands)) != len(self.quantile_bands):
            raise ValueError("quantile_bands must be unique")


@dataclass(frozen=True, kw_only=True, slots=True)
class ShapConfig(_Config):
    """Configure SHAP sampling and displayed feature contributions.

    Parameters
    ----------
    max_samples : int, default 10000
        Positive maximum row count sampled without replacement from validation.
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


@dataclass(frozen=True, kw_only=True, slots=True)
class CommonPlotsConfig(_Config):
    """Configure figures applicable to all supported diagnostic tasks.

    Parameters
    ----------
    figure : FigureConfig
        Shared export settings; defaults to a fresh FigureConfig.
    psi_scores : bool, default True
        Draw feature PSI bars when PSI calculation is enabled.
    psi_distribution_features : tuple[str, ...] or None, default None
        None draws monthly levels for every PSI feature; empty draws none.
        Explicit names must be included in the PSI feature configuration.
    numerical_distributions : NumericalFeatureDistributionConfig or None
        Defaults to all numerical features. None disables these plots.
    shap : ShapConfig or None, default None
        Optional SHAP diagnostics; None disables SHAP calculation and plots.
    """

    figure: FigureConfig = field(default_factory=FigureConfig)
    psi_scores: bool = True
    psi_distribution_features: tuple[str, ...] | None = None
    numerical_distributions: NumericalFeatureDistributionConfig | None = field(
        default_factory=NumericalFeatureDistributionConfig
    )
    shap: ShapConfig | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class ClassificationPlotsConfig(_Config):
    """Choose supervised binary-classification figures.

    Parameters
    ----------
    common : CommonPlotsConfig
        Shared figures and export settings; defaults to CommonPlotsConfig.
    precision_recall : bool, default True
        Display the unweighted precision-recall curve.
    f1_threshold : bool, default True
        Display unweighted F1 across distinct score thresholds.
    calibration : bool, default True
        Display unweighted observed rates against predicted probabilities.
    gains : bool, default True
        Display exposure-weighted cumulative gains for nonnegative outcomes.
    lift : bool, default True
        Display exposure-weighted cumulative lift.
    monthly_observed_fitted : bool, default True
        Display monthly exposure-weighted observed and fitted means.
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
    monthly_observed_fitted: bool = True
    max_curve_points: int = 1_000
    calibration_bins: int = 10
    histogram_bins: int = 30


@dataclass(frozen=True, kw_only=True, slots=True)
class RegressionPlotsConfig(_Config):
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
    monthly_observed_fitted : bool, default True
        Display monthly exposure-weighted observed and fitted means.

    Notes
    -----
    Ranking plots default off so ordinary signed regression targets are valid.
    Frequency and severity recipes can enable them explicitly.
    """

    common: CommonPlotsConfig = field(default_factory=CommonPlotsConfig)
    gains: bool = False
    lift: bool = False
    monthly_observed_fitted: bool = True


@dataclass(frozen=True, kw_only=True, slots=True)
class PUPlotsConfig(_Config):
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


def _diagnostics_check(config):
    _Config.__post_init__(config)
    if isinstance(config.folds, tuple):
        if (
            not config.folds
            or any(value < 1 for value in config.folds)
            or len(set(config.folds)) != len(config.folds)
        ):
            raise ValueError("folds must contain unique positive one-based indices")
    if config.plots is not None and config.psi is None:
        common = config.plots.common
        if common.psi_distribution_features:
            raise ValueError(
                "Explicit PSI distribution features require PSI calculation"
            )


@dataclass(frozen=True, kw_only=True, slots=True)
class ClassificationDiagnosticsConfig(_Config):
    """Configure supervised binary-classification diagnostics.

    Parameters
    ----------
    folds : {'all', 'last'} or tuple[int, ...], default 'last'
        Evaluated folds to diagnose. Explicit indices are unique, positive,
        one-based original fold numbers and must not refer to skipped folds.
    psi : PSIOptions or None
        PSI tables and fitted reference levels; defaults to PSIOptions.
        None disables PSI calculation and related plots.
    plots : ClassificationPlotsConfig or None
        Defaults to ClassificationPlotsConfig. None disables every figure
        while leaving PSI tables enabled.
    """

    folds: Literal["all", "last"] | tuple[int, ...] = "last"
    psi: PSIOptions | None = field(default_factory=PSIOptions)
    plots: ClassificationPlotsConfig | None = field(
        default_factory=ClassificationPlotsConfig
    )
    __post_init__ = _diagnostics_check

    @property
    def task(self) -> Literal["classification"]:
        """Return the fixed supervised classification task identifier."""
        return "classification"


@dataclass(frozen=True, kw_only=True, slots=True)
class RegressionDiagnosticsConfig(_Config):
    """Configure regression-family diagnostics.

    Parameters
    ----------
    task : {'regression', 'frequency', 'severity'}, default 'regression'
        Regression task label for validation and diagnostic metadata.
    folds : {'all', 'last'} or tuple[int, ...], default 'last'
        Evaluated folds to diagnose; explicit indices are unique one-based
        original fold numbers and must not refer to skipped folds.
    psi : PSIOptions or None
        Defaults to PSIOptions; None disables PSI calculation and its figures.
    plots : RegressionPlotsConfig or None
        Defaults to RegressionPlotsConfig; None disables all figures.
    """

    task: Literal["regression", "frequency", "severity"] = "regression"
    folds: Literal["all", "last"] | tuple[int, ...] = "last"
    psi: PSIOptions | None = field(default_factory=PSIOptions)
    plots: RegressionPlotsConfig | None = field(default_factory=RegressionPlotsConfig)
    __post_init__ = _diagnostics_check


@dataclass(frozen=True, kw_only=True, slots=True)
class PUDiagnosticsConfig(_Config):
    """Configure positive-unlabeled diagnostics.

    Parameters
    ----------
    folds : {'all', 'last'} or tuple[int, ...], default 'last'
        Evaluated folds to diagnose; explicit indices are unique one-based
        original fold numbers and must not refer to skipped folds.
    psi : PSIOptions or None
        Defaults to PSIOptions; None disables PSI calculation and its figures.
    plots : PUPlotsConfig or None
        Defaults to PUPlotsConfig; None disables all figures. Supervised plot
        configurations are rejected during construction.
    """

    folds: Literal["all", "last"] | tuple[int, ...] = "last"
    psi: PSIOptions | None = field(default_factory=PSIOptions)
    plots: PUPlotsConfig | None = field(default_factory=PUPlotsConfig)
    __post_init__ = _diagnostics_check

    @property
    def task(self) -> Literal["pu_classification"]:
        """Return the fixed positive-unlabeled classification identifier."""
        return "pu_classification"


DiagnosticsConfig = (
    ClassificationDiagnosticsConfig | RegressionDiagnosticsConfig | PUDiagnosticsConfig
)


@dataclass(frozen=True, kw_only=True, slots=True)
class TrackingConfig(_Config):
    """Configure model packaging in an externally configured MLflow store.

    Parameters
    ----------
    log_trial_models : bool, default False
        Package every fold model in each tuning trial.
    log_fold_models : bool, default False
        Package each model in the evaluation CV pass.
    log_final_model : bool, default True
        Package the model fitted on all retained rows.
    input_example_rows : int, default 5
        Positive maximum serving-example row count, capped at sample size.
    model_name : str, default 'model'
        Nonempty MLflow model artifact name shared across nested runs.
    registered_model_name : str or None, default None
        Registry name used only for final fitting, never trial or CV models.
    conda_env : pathlib.Path or None, default None
        Existing serving environment file; None pins installed dependencies.

    Notes
    -----
    Metrics, run metadata, and configuration artifacts are always logged.
    The caller owns tracking URI, experiment selection, and Python logging.
    """

    log_trial_models: bool = False
    log_fold_models: bool = False
    log_final_model: bool = True
    input_example_rows: int = 5
    model_name: str = "model"
    registered_model_name: str | None = None
    conda_env: Path | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class PostFitEvaluationConfig(_Config):
    """Control calls to a separately supplied custom post-fit evaluator.

    Parameters
    ----------
    stages : tuple[{'tuning', 'cv', 'final'}, ...], default ('cv', 'final')
        Unique enabled stages. An empty tuple disables all calls.
    splits : tuple[{'train', 'val'}, ...], default ('train', 'val')
        Unique enabled splits. Final fitting has train only. Empty disables
        all calls. No independent test split exists in this workflow.
    write_artifacts : bool, default True
        Call the evaluator's optional image writer after scalar evaluation.
        False disables images while retaining enabled scalar metrics.

    Notes
    -----
    The evaluator is not copied or reset. Custom scores are logged separately
    and do not become CV averages or optimization objectives.
    """

    stages: tuple[Literal["tuning", "cv", "final"], ...] = ("cv", "final")
    splits: tuple[Literal["train", "val"], ...] = ("train", "val")
    write_artifacts: bool = True

    def __post_init__(self):
        _Config.__post_init__(self)
        if len(set(self.stages)) != len(self.stages) or len(set(self.splits)) != len(
            self.splits
        ):
            raise ValueError("post-fit stages and splits must be unique")


@dataclass(frozen=True, kw_only=True, slots=True)
class TrainingConfig(_Config):
    """Compose immutable settings for the complete model-training workflow.

    Parameters
    ----------
    cv : CrossValidationConfig
        Required chronological validation settings, reused by every trial.
    feature_selection : RFEFeatureSelectionConfig or None, default None
        Optional early batch RFE; requires feature_ranker.
        None retains every feature.
    tuning : TuningConfig or None, default None
        Optional Optuna search; requires a separately supplied param_space.
    final_fit : FinalFitConfig
        Final iteration policy; defaults to FinalFitConfig.
    diagnostics : DiagnosticsConfig or None, default None
        Task-specific PSI and figures for selected evaluation folds.
    tracking : TrackingConfig
        Model packaging settings; defaults to TrackingConfig.
    post_fit : PostFitEvaluationConfig
        Custom evaluator stage/split policy; defaults to PostFitEvaluationConfig.

    Notes
    -----
    Data, native model params, model_factory, metric, feature_ranker,
    param_space, evaluator and run_name remain explicit function arguments.
    Configurations contain no executing dependencies. All fields are
    keyword-only, validated on creation,
    and immutable; dataclasses.replace creates modified configurations.

    Examples
    --------
    >>> config = TrainingConfig(cv=CrossValidationConfig(n_splits=3))
    >>> config.feature_selection is None
    True
    """

    cv: CrossValidationConfig
    feature_selection: RFEFeatureSelectionConfig | None = None
    tuning: TuningConfig | None = None
    final_fit: FinalFitConfig = field(default_factory=FinalFitConfig)
    diagnostics: DiagnosticsConfig | None = None
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    post_fit: PostFitEvaluationConfig = field(default_factory=PostFitEvaluationConfig)

    def __post_init__(self):
        _Config.__post_init__(self)
        if self.diagnostics is not None and isinstance(self.diagnostics.folds, tuple):
            if any(
                not self.cv.skip_splits < index <= self.cv.n_splits
                for index in self.diagnostics.folds
            ):
                raise ValueError("Diagnostic folds must be evaluated CV fold indices")


__all__ = [
    "RFEFeatureSelectionConfig",
    "CrossValidationConfig",
    "TuningConfig",
    "FinalFitConfig",
    "PSIOptions",
    "FigureConfig",
    "NumericalFeatureDistributionConfig",
    "ShapConfig",
    "CommonPlotsConfig",
    "ClassificationPlotsConfig",
    "RegressionPlotsConfig",
    "PUPlotsConfig",
    "ClassificationDiagnosticsConfig",
    "RegressionDiagnosticsConfig",
    "PUDiagnosticsConfig",
    "DiagnosticsConfig",
    "TrackingConfig",
    "PostFitEvaluationConfig",
    "TrainingConfig",
]
