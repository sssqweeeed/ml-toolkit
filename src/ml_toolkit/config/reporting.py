"""Shared reporting settings; individual reports own their parameters."""

from dataclasses import dataclass, field
from math import isfinite
from typing import Literal

from .._time import PERIOD_INTERVALS
from ..feature_levels import FeatureLevelConfig
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
class ReportingConfig:
    """Enable reporting by training stage and configure PSI, SHAP and exports.

    Stage switches gate prediction evaluators, data analyzers, PSI and SHAP,
    including all additional metrics and artifacts, not training
    metrics or optimization. All validation folds use the same settings.
    PSI and observed/fitted use feature levels fitted on training rows.
    ``features=None`` infers all input features; ``()`` disables level reports.
    ``period`` and ``epsilon`` apply only to PSI.

    Parameters
    ----------
    on_tuning : bool, default False
        Enable additional reports for Optuna trials.
    on_validation : bool, default True
        Enable additional reports for every completed validation fold.
    on_test : bool, default True
        Enable additional reports for the model evaluated on held-out test data.
    on_final_fit : bool, default True
        Enable applicable train-only reports after fitting all eligible data.
    psi : bool, default False
        Enable data analysis of population stability; uses no predictions.
    shap : ShapConfig or None, default None
        Optional native-model explanations of held-out rows.
    features : tuple[FeatureLevelConfig, ...] or None, default None
        Shared feature definitions; None infers columns, () disables levels.
    n_bins : int, default 10
        Positive number of train-fitted numerical quantile bins.
    epsilon : float, default 1e-6
        PSI share floor strictly between zero and one.
    period : TimePeriod, default "month"
        Calendar period for temporal PSI distributions.
    figure : FigureConfig
        Shared image dimensions, resolution and output format.
    """

    on_tuning: bool = False
    on_validation: bool = True
    on_test: bool = True
    on_final_fit: bool = True
    psi: bool = False
    shap: ShapConfig | None = None
    features: tuple[FeatureLevelConfig, ...] | None = None
    n_bins: int = 10
    epsilon: float = 1e-6
    period: TimePeriod = "month"
    figure: FigureConfig = field(default_factory=FigureConfig)

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
