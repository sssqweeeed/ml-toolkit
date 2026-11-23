"""The scalar metric contract shared by training and custom metrics."""

from dataclasses import dataclass
from typing import Protocol

from ..data import Pool, validate_prediction
from ..types import FloatArray


class MetricResult(Protocol):
    """Describe a metric result exporting named scalar scores through to_dict.

    Custom results may be dataclasses or other objects implementing to_dict;
    inheriting from this protocol is optional.
    """

    def to_dict(self) -> dict[str, float]:
        """Return named scores for reporting and objective selection.

        Returns
        -------
        dict[str, float]
            Scores in their natural units; undefined non-objective scores may be NaN.
        """
        ...


class Metric(Protocol):
    """Describe a training metric accepting targets, predictions, and weights.

    Implement the three positional arguments and return a MetricResult. The
    training data boundary validates alignment and nonnegative weights; a
    custom metric is responsible for its own mathematical domain constraints.
    """

    def __call__(
        self, y: FloatArray, prediction: FloatArray, weight: FloatArray, /
    ) -> MetricResult:
        """Return scalar results for one dataset.

        Parameters
        ----------
        y, prediction, weight : numpy.ndarray
            Aligned one-dimensional targets, predictions and nonnegative weights.

        Returns
        -------
        MetricResult
            Result supplying to_dict().
        """
        ...


@dataclass(frozen=True)
class MetricScores:
    """Wrap named scores produced by a project-owned metric.

    Parameters
    ----------
    scores : dict[str, float]
        Named scalar results. The dictionary is copied when exporting scores.

    Examples
    --------
    >>> MetricScores({"mae": 0.5}).to_dict()
    {'mae': 0.5}
    """

    scores: dict[str, float]

    def to_dict(self) -> dict[str, float]:
        """Return a copy of the named scores.

        Returns
        -------
        dict[str, float]
            Scalar scores without sharing the stored dictionary.
        """
        return self.scores.copy()


def calculate_metrics(
    metric: Metric, data: Pool, prediction: FloatArray
) -> dict[str, float]:
    """Score one aligned prediction on a pool with a metric.

    Parameters
    ----------
    metric : Metric
        Callable receiving targets, predictions and weights.
    data : Pool
        Rows supplying targets and weights.
    prediction : numpy.ndarray
        One finite prediction per pool row.

    Returns
    -------
    dict[str, float]
        The metric's named scores.

    Raises
    ------
    ValueError
        If the prediction is misaligned or not finite.
    """
    validate_prediction(data, prediction)
    return metric(
        data.y.to_numpy().astype(float),
        prediction,
        data.weight.to_numpy().astype(float),
    ).to_dict()
