from dataclasses import dataclass

import numpy as np
from sklearn.metrics import d2_tweedie_score, mean_absolute_error, mean_poisson_deviance

from ..types import FloatArray
from .ranking import ranking_metrics


def root_mean_squared_error(
    y_target: FloatArray,
    y_pred: FloatArray,
    weight: FloatArray,
) -> float:
    return float(np.sqrt(np.average(np.square(y_target - y_pred), weights=weight)))


@dataclass(frozen=True)
class MetricsFrequncy:
    """Store exposure-weighted frequency errors and ranking scores.

    Attributes
    ----------
    mean_poisson_deviance : float
        Exposure-weighted Poisson deviance.
    d2_poisson_score : float
        Fraction of Poisson deviance explained relative to a constant
        prediction.
    mean_absolute_error : float
        Exposure-weighted mean absolute error.
    root_mean_squared_error : float
        Square root of exposure-weighted mean squared error.
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

    Notes
    -----
    MetricsFrequency is the preferred public alias of this class.
    """

    mean_poisson_deviance: float
    d2_poisson_score: float
    mean_absolute_error: float
    root_mean_squared_error: float
    prediction_over_observed: float
    lift_at_10: float
    gain_at_10: float
    normalized_gini: float

    def to_dict(self) -> dict[str, float]:
        """Return named scalar scores.

        Returns
        -------
        dict[str, float]
            All scores in this result.
        """
        return {
            "mean_poisson_deviance": self.mean_poisson_deviance,
            "d2_poisson_score": self.d2_poisson_score,
            "mean_absolute_error": self.mean_absolute_error,
            "root_mean_squared_error": self.root_mean_squared_error,
            "prediction_over_observed": self.prediction_over_observed,
            "lift_at_10": self.lift_at_10,
            "gain_at_10": self.gain_at_10,
            "normalized_gini": self.normalized_gini,
        }


def metrics_frequncy(
    y_target: FloatArray, y_pred: FloatArray, exposure: FloatArray
) -> MetricsFrequncy:
    """Calculate exposure-weighted Poisson and ranking metrics.

    Parameters
    ----------
    y_target : numpy.ndarray of shape (n_samples,)
        Nonnegative observed rates with positive exposure-weighted total.
    y_pred : numpy.ndarray of shape (n_samples,)
        Finite, strictly positive predicted rates.
    exposure : numpy.ndarray of shape (n_samples,)
        Finite, nonnegative observation exposures with positive total.

    Returns
    -------
    MetricsFrequency
        Poisson deviance and D2, absolute and squared errors, prediction ratio,
        and ranking at 10% of exposure.

    Raises
    ------
    ValueError
        Ranking inputs are invalid or targets and predictions violate Poisson
        metric requirements.

    Notes
    -----
    Targets and predictions are rates, not exposure-multiplied totals.
    Exposures weight error scores and determine ranking capacity.
    metrics_frequency is the preferred public alias of this function.

    Examples
    --------
    >>> import numpy as np
    >>> from ml_toolkit.metrics import metrics_frequency
    >>> scores = metrics_frequency(
    ...     np.array([0.0, 1.0, 2.0]),
    ...     np.array([0.2, 1.0, 1.8]),
    ...     np.ones(3),
    ... )
    >>> round(scores.prediction_over_observed, 2)
    1.0
    """
    ranking = ranking_metrics(
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
        root_mean_squared_error=root_mean_squared_error(y_target, y_pred, exposure),
        prediction_over_observed=float(
            (y_pred * exposure).sum() / (y_target * exposure).sum()
        ),
        lift_at_10=ranking.selected_points[0].lift,
        gain_at_10=ranking.selected_points[0].gain,
        normalized_gini=ranking.normalized_gini,
    )


MetricsFrequency = MetricsFrequncy
metrics_frequency = metrics_frequncy
