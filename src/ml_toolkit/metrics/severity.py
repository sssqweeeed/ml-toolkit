from dataclasses import dataclass

import numpy as np
from sklearn.metrics import d2_tweedie_score, mean_tweedie_deviance

from ..types import FloatArray
from .ranking import ranking_metrics
from .regression import MetricsRegression, metrics_regression


@dataclass(frozen=True)
class MetricsSeverity(MetricsRegression):
    """Store exposure-weighted severity errors and ranking scores.

    Attributes
    ----------
    mean_tweedie_deviance : float
        Exposure-weighted deviance for the requested Tweedie power.
    d2_tweedie_score : float
        Fraction of Tweedie deviance explained relative to a constant
        prediction.
    mean_absolute_error : float
        Exposure-weighted mean absolute error.
    root_mean_squared_error : float
        Square root of exposure-weighted mean squared error.
    mean_squared_error : float
        Exposure-weighted mean squared error.
    r2_score : float
        Weighted coefficient of determination, as in metrics_regression.
    mean_error : float
        Exposure-weighted prediction minus target; positive means overprediction.
    prediction_over_observed : float
        Total exposure-weighted prediction divided by the observed total.
    lift_at_10 : float
        Gain at the highest-scored 10% of exposure divided by 0.1.
    gain_at_10 : float
        Share of total observed outcome captured by the highest-scored 10% of
        exposure.
    normalized_gini : float
        Model Gini divided by perfect-ranking Gini; NaN when the latter is
        close to zero.
    """

    mean_tweedie_deviance: float
    d2_tweedie_score: float
    prediction_over_observed: float
    lift_at_10: float
    gain_at_10: float
    normalized_gini: float


def metrics_severity(
    y_target: FloatArray,
    y_pred: FloatArray,
    exposure: FloatArray,
    power: float,
) -> MetricsSeverity:
    """Calculate exposure-weighted Tweedie and ranking metrics.

    Parameters
    ----------
    y_target : numpy.ndarray of shape (n_samples,)
        Finite observed severity values satisfying the chosen Tweedie power,
        with positive exposure-weighted total.
    y_pred : numpy.ndarray of shape (n_samples,)
        Finite predicted severities satisfying the chosen Tweedie power.
    exposure : numpy.ndarray of shape (n_samples,)
        Finite, nonnegative row weights with positive total.
    power : float
        Tweedie variance power passed to scikit-learn; for example, 2 uses
        Gamma deviance and requires strictly positive targets and predictions.

    Returns
    -------
    MetricsSeverity
        Tweedie deviance and D2, error scores, prediction ratio, and ranking at
        10% of exposure.

    Raises
    ------
    ValueError
        Ranking inputs or the Tweedie domain are invalid.

    Notes
    -----
    Use functools.partial to fix power when passing this function as a training
    metric. Ranking scores use the same exposure weights as the error scores.

    Examples
    --------
    >>> import numpy as np
    >>> from functools import partial
    >>> from ml_toolkit.metrics import metrics_severity
    >>> gamma_metric = partial(metrics_severity, power=2.0)
    >>> scores = gamma_metric(
    ...     np.array([100.0, 200.0, 300.0]),
    ...     np.array([120.0, 180.0, 240.0]),
    ...     np.array([1.0, 2.0, 3.0]),
    ... )
    >>> scores.mean_absolute_error
    40.0
    >>> round(scores.mean_tweedie_deviance, 4)
    0.0359
    >>> round(scores.prediction_over_observed, 4)
    0.8571

    In this calculation example, amounts are dollars and weights are claim
    counts. The weighted absolute error is $40 per claim; total predicted
    cost is about 14.3% below observed cost. Gamma deviance is dimensionless.
    """
    observed = np.sum(y_target * exposure)
    predicted = np.sum(y_pred * exposure)

    ranking = ranking_metrics(
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
        **metrics_regression(y_target, y_pred, exposure).to_dict(),
        prediction_over_observed=float(
            predicted / observed if observed != 0 else np.nan
        ),
        lift_at_10=ranking.selected_points[0].lift,
        gain_at_10=ranking.selected_points[0].gain,
        normalized_gini=ranking.normalized_gini,
    )
