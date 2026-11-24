"""Prediction, model and data reports, each with its computation and figures."""

from .base import (
    DataAnalyzer,
    ModelEvaluator,
    PredictionEvaluator,
    comparison_pair,
    data_comparison_pair,
    validate_inputs,
    validate_report_metrics,
)
from .classification import Calibration, F1Threshold, PrecisionRecall
from .distributions import NumericalFeatureDistributions
from .grouped import GroupedMetrics
from .metric_trend import MetricTrend
from .permutation import PermutationImportance
from .pu import PUPositiveCapture, PUScoreDistribution
from .ranking import Gains, Lift
from .regression import ObservedFitted, TimeObservedFitted
from .residuals import ResidualResult, Residuals
from .robustness import RobustnessCurve
from .shrink import ShrinkCurve

__all__ = [
    "Calibration",
    "DataAnalyzer",
    "F1Threshold",
    "Gains",
    "GroupedMetrics",
    "Lift",
    "MetricTrend",
    "ModelEvaluator",
    "NumericalFeatureDistributions",
    "ObservedFitted",
    "PUPositiveCapture",
    "PUScoreDistribution",
    "PermutationImportance",
    "PrecisionRecall",
    "PredictionEvaluator",
    "ResidualResult",
    "Residuals",
    "RobustnessCurve",
    "ShrinkCurve",
    "TimeObservedFitted",
    "comparison_pair",
    "data_comparison_pair",
    "validate_inputs",
    "validate_report_metrics",
]
