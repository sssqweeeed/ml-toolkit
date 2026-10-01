from dataclasses import dataclass

import numpy as np
from sklearn.metrics import d2_tweedie_score, mean_absolute_error, mean_tweedie_deviance

from .ranking import _ranking_metrics
from .frequency import _root_mean_squared_error

@dataclass(frozen=True)
class MetricsSeverity:
    mean_tweedie_deviance: float
    d2_tweedie_score: float
    mean_absolute_error: float
    root_mean_squared_error: float
    prediction_over_observed: float
    lift_at_10: float
    gain_at_10: float
    normalized_gini: float

def metrics_severity(
    y_target: np.ndarray,
    y_pred: np.ndarray,
    exposure: np.ndarray,
    power: float,
) -> MetricsSeverity:
    observed = np.sum(y_target * exposure)
    predicted = np.sum(y_pred * exposure)

    ranking_metrics = _ranking_metrics(
        y_target=y_target, y_pred=y_pred, exposure=exposure, selected_percents=(10.0,)
    )
    return MetricsSeverity(
        mean_tweedie_deviance=float(
            mean_tweedie_deviance(
                y_target,
                y_pred,
                sample_weight=exposure,
                power=power,
            )
        ),
        d2_tweedie_score=float(
            d2_tweedie_score(
                y_target,
                y_pred,
                sample_weight=exposure,
                power=power,
            )
        ),
        mean_absolute_error=float(
            mean_absolute_error(
                y_target,
                y_pred,
                sample_weight=exposure,
            )
        ),
        root_mean_squared_error=float(
            _root_mean_squared_error(
                y_target,
                y_pred,
                weight=exposure,
            )
        ),
        prediction_over_observed=float(
            predicted / observed if observed != 0 else np.nan
        ),
        lift_at_10=ranking_metrics.selected_points[0].lift,
        gain_at_10=ranking_metrics.selected_points[0].gain,
        normalized_gini=ranking_metrics.normalized_gini,
    )
