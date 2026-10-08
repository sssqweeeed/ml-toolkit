"""The scalar metric contract shared by training and custom metrics."""

from dataclasses import dataclass
from typing import Protocol

from ..types import FloatArray


class MetricResult(Protocol):
    """Describe a result with named scalar scores."""

    def to_dict(self) -> dict[str, float]:
        """Return named scores for reporting and objective selection.

        Returns
        -------
        dict[str, float]
            Scores in their natural units; undefined non-objective scores may be NaN.
        """
        ...


class Metric(Protocol):
    """Describe a callable evaluating aligned targets, predictions and weights."""

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
