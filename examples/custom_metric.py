"""A project-owned metric shared by training and adapter examples."""

import numpy as np

from ml_toolkit.metrics import MetricScores
from ml_toolkit.types import FloatArray


def weighted_mae(
    observed: FloatArray, prediction: FloatArray, weight: FloatArray
) -> MetricScores:
    """Return weighted absolute error in the target's units under ``mae``.

    Parameters
    ----------
    observed, prediction, weight : FloatArray
        Aligned one-dimensional values. Weights must be finite, nonnegative,
        and have positive total; training validates that data boundary.

    Returns
    -------
    MetricScores
        Named scalar error usable for reporting and minimization.
    """
    error = abs(observed - prediction)
    return MetricScores({"mae": float(np.average(error, weights=weight))})
