"""Prediction reports grouped by task, with stable public imports."""

from .base import PredictionEvaluator
from .classification import Calibration, F1Threshold, PrecisionRecall
from .grouped import GroupedMetrics
from .metric_trend import MetricTrend
from .pu import PUPositiveCapture, PUScoreDistribution
from .ranking import Gains, Lift
from .regression import ObservedFitted, TimeObservedFitted
from .residuals import ResidualResult, Residuals
from ._compat import NumericalFeatureDistributions
from ._common import (
    comparison_pair as comparison_pair,
    validate_inputs as validate_inputs,
)

__all__ = [
    "PredictionEvaluator",
    "MetricTrend",
    "Residuals",
    "ResidualResult",
    "PrecisionRecall",
    "F1Threshold",
    "Calibration",
    "Gains",
    "Lift",
    "PUScoreDistribution",
    "PUPositiveCapture",
    "ObservedFitted",
    "TimeObservedFitted",
    "GroupedMetrics",
    "NumericalFeatureDistributions",
]
