from dataclasses import dataclass

import numpy as np
from sklearn.metrics import d2_tweedie_score, mean_absolute_error, mean_poisson_deviance

from .ranking import _ranking_metrics

def _root_mean_squared_error(
    y_target: np.ndarray,
    y_pred: np.ndarray,
    weight: np.ndarray,
) -> float:
    return float(np.sqrt(np.average(np.square(y_target - y_pred), weights=weight)))

@dataclass(frozen=True)
class MetricsFrequncy:
    mean_poisson_deviance: float
    d2_poisson_score: float
    mean_absolute_error: float
    root_mean_squared_error: float
    prediction_over_observed: float
    lift_at_10: float
    gain_at_10: float
    normalized_gini: float

def metrics_frequncy(
    y_target: np.ndarray, y_pred: np.ndarray, exposure: np.ndarray
) -> MetricsFrequncy:
    ranking_metrics = _ranking_metrics(
        y_target=y_target, y_pred=y_pred, exposure=exposure, selected_percents=(10.0,)
    )
    return MetricsFrequncy(
        mean_poisson_deviance=float(
            mean_poisson_deviance(y_target, y_pred, sample_weight=exposure)
        ),
        d2_poisson_score=float(
            d2_tweedie_score(y_target, y_pred, sample_weight=exposure, power=1)
        ),
        mean_absolute_error=float(
            mean_absolute_error(y_target, y_pred, sample_weight=exposure)
        ),
        root_mean_squared_error=_root_mean_squared_error(y_target, y_pred, exposure),
        prediction_over_observed=float(
            (y_pred * exposure).sum() / (y_target * exposure).sum()
        ),
        lift_at_10=ranking_metrics.selected_points[0].lift,
        gain_at_10=ranking_metrics.selected_points[0].gain,
        normalized_gini=ranking_metrics.normalized_gini,
    )

MetricsFrequency = MetricsFrequncy
metrics_frequency = metrics_frequncy
