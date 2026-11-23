"""Shared reporting settings; individual reports own their parameters."""

from dataclasses import dataclass, field

from .._time import PERIOD_INTERVALS
from ..explainability import ShapConfig
from ..feature_levels import FeatureLevelConfig
from ..figures import FigureConfig
from ..types import TimePeriod


@dataclass(frozen=True, kw_only=True, slots=True)
class ReportingConfig:
    """Enable reporting by training stage and configure PSI, SHAP and exports.

    Stage switches gate prediction/model evaluators, data analyzers, PSI and SHAP,
    including their additional metrics and report artifacts, not training
    metrics or optimization. PSI baseline packaging follows model logging flags
    independently of stage report switches. All validation folds use the same settings.
    An evaluator selected by TuningConfig still calculates objective metrics on
    every trial; these switches gate its reports only when it is also explicitly
    supplied through prediction_evaluators.
    PSI and observed/fitted use feature levels fitted on training rows.
    ``features=None`` infers model and auxiliary features; ``()`` disables level reports.
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
        Enable population stability reports and baseline packaging with saved
        models; uses no predictions. features=() disables the baseline too.
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
