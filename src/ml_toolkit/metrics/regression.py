"""General weighted regression scores, including signed targets."""

from dataclasses import asdict, dataclass

import numpy as np

from ..types import FloatArray


@dataclass(frozen=True)
class MetricsRegression:
    """Store weighted regression scores.

    Attributes
    ----------
    mean_absolute_error : float
        Weighted mean absolute prediction error.
    mean_squared_error : float
        Weighted mean squared prediction error.
    root_mean_squared_error : float
        Square root of the weighted mean squared error.
    r2_score : float
        Fraction of weighted target variance explained by predictions.
    mean_error : float
        Weighted mean of prediction minus target.

    Notes
    -----
    ``mean_error`` is prediction minus target: positive values indicate
    overprediction. R² is NaN with fewer than two positive-weight rows;
    constant targets score 1 for perfect predictions and 0 otherwise.
    """

    mean_absolute_error: float
    mean_squared_error: float
    root_mean_squared_error: float
    r2_score: float
    mean_error: float

    def to_dict(self) -> dict[str, float]:
        """Return all named scores, including specialized subclass scores."""
        return asdict(self)


def metrics_regression(
    y_target: FloatArray,
    y_pred: FloatArray,
    sample_weight: FloatArray | None = None,
) -> MetricsRegression:
    """Calculate regression errors for aligned finite one-dimensional arrays.

    Parameters
    ----------
    y_target, y_pred : numpy.ndarray of shape (n_samples,)
        Nonempty finite targets and predictions, including signed values.
    sample_weight : numpy.ndarray of shape (n_samples,) or None, default None
        Finite nonnegative weights with positive total; None uses equal weights.

    Returns
    -------
    MetricsRegression
        Weighted absolute, squared and signed errors, and R².

    Raises
    ------
    ValueError
        Arrays are empty, misaligned, nonfinite, or weights are invalid.
    """
    target = np.asarray(y_target, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    weight = (
        np.ones_like(target)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )
    if (
        target.ndim != 1
        or target.size == 0
        or prediction.shape != target.shape
        or weight.shape != target.shape
    ):
        raise ValueError(
            "Targets, predictions and weights must be nonempty aligned 1D arrays"
        )
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("Targets and predictions must be finite")
    if not np.isfinite(weight).all() or (weight < 0).any() or weight.sum() <= 0:
        raise ValueError("Weights must be finite, nonnegative, and have positive total")

    positive = weight > 0
    target, prediction, weight = (
        target[positive],
        prediction[positive],
        weight[positive],
    )
    error = prediction - target
    mse = float(np.average(error**2, weights=weight))
    if len(target) < 2:
        r2 = float("nan")
    elif (target == target[0]).all():
        r2 = 1.0 if (error == 0).all() else 0.0
    else:
        variance = float(
            np.average(
                (target - np.average(target, weights=weight)) ** 2, weights=weight
            )
        )
        r2 = 1.0 - mse / variance
    return MetricsRegression(
        mean_absolute_error=float(np.average(np.abs(error), weights=weight)),
        mean_squared_error=mse,
        root_mean_squared_error=float(np.sqrt(mse)),
        r2_score=r2,
        mean_error=float(np.average(error, weights=weight)),
    )
