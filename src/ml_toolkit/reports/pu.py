"""Reports for known-positive and unlabeled predictions."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from numpy.typing import ArrayLike

from ..data import Pool
from ..feature_levels import FeatureLevelConfig
from ..figures import FigureConfig, saved_figure
from ..metrics.pu import positive_capture_curve, pu_arrays
from ..types import FloatArray
from .base import PredictionEvaluator, write_heldout_plot

__all__ = [
    "PUPositiveCapture",
    "PUScoreDistribution",
    "positive_capture",
    "score_distribution",
]


def _pu_distribution(
    observed: FloatArray,
    prediction: FloatArray,
    *,
    weight: FloatArray,
    save_fig: Path,
    figure: FigureConfig,
    histogram_bins: int,
) -> None:
    score_distribution(
        observed,
        prediction,
        weights=weight,
        save_fig=save_fig,
        figure=figure,
        histogram_bins=histogram_bins,
    )


def _pu_capture(
    observed: FloatArray,
    prediction: FloatArray,
    *,
    weight: FloatArray,
    save_fig: Path,
    figure: FigureConfig,
) -> None:
    positive_capture(
        observed, prediction, weights=weight, save_fig=save_fig, figure=figure
    )


@dataclass(frozen=True, kw_only=True)
class PUScoreDistribution(PredictionEvaluator):
    """Compare score distributions for known positives and unlabeled rows.

    Parameters
    ----------
    name : str, default 'pu_score_distribution'
        Unique report name and artifact namespace.
    histogram_bins : int, default 30
        Number of score bins.

    Notes
    -----
    Labels 1 and 0 mean known positive and unlabeled. Each group's weighted
    histogram is normalized separately; both need positive total weight.
    Uses validation or test rows and saves
    ``<split>/pu_score_distribution.<format>``.
    """

    name: str = "pu_score_distribution"
    histogram_bins: int = 30

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.histogram_bins < 1:
            raise ValueError("histogram_bins must be positive")

    def parameters(self) -> dict[str, object]:
        return {"histogram_bins": self.histogram_bins}

    def write_artifacts(
        self,
        *,
        data: Pool,
        prediction: FloatArray,
        split: pl.Series,
        artifact_dir: Path,
        figure: FigureConfig,
        features: tuple[FeatureLevelConfig, ...] | None,
        n_bins: int,
    ) -> list[Path]:
        return write_heldout_plot(
            data,
            prediction,
            split,
            artifact_dir,
            figure,
            "pu_score_distribution",
            _pu_distribution,
            histogram_bins=self.histogram_bins,
        )


@dataclass(frozen=True, kw_only=True)
class PUPositiveCapture(PredictionEvaluator):
    """Plot known-positive capture against unlabeled review capacity.

    Parameters
    ----------
    name : str, default 'pu_positive_capture'
        Unique report name and artifact namespace.

    Notes
    -----
    Labels 1 and 0 mean known positive and unlabeled. Both axes use weighted
    percentages; tied scores enter together. Capture is not recall for all
    positives without assumptions about labeling. Uses validation or test rows
    and saves ``<split>/pu_positive_capture.<format>``.
    """

    name: str = "pu_positive_capture"

    def write_artifacts(
        self,
        *,
        data: Pool,
        prediction: FloatArray,
        split: pl.Series,
        artifact_dir: Path,
        figure: FigureConfig,
        features: tuple[FeatureLevelConfig, ...] | None,
        n_bins: int,
    ) -> list[Path]:
        return write_heldout_plot(
            data,
            prediction,
            split,
            artifact_dir,
            figure,
            "pu_positive_capture",
            _pu_capture,
        )


def score_distribution(
    labels: ArrayLike,
    probabilities: ArrayLike,
    save_fig: str | Path,
    *,
    weights: ArrayLike | None = None,
    histogram_bins: int = 30,
    figure: FigureConfig | None = None,
) -> None:
    """Save score distributions for known positives and unlabeled rows.

    Parameters
    ----------
    labels : array-like of shape (n_samples,)
        1 identifies a known positive; 0 identifies an unlabeled observation.
    probabilities : array-like of shape (n_samples,)
        Finite model scores in [0, 1].
    save_fig : str or pathlib.Path
        Output figure path; parent directories are created.
    weights : array-like or None, default None
        Finite, nonnegative aligned weights with positive total in each group;
        None uses equal weights.

    figure : FigureConfig or None, default None
        Optional export dimensions, DPI, and format. None preserves native
        dimensions and the requested path format.
    histogram_bins : int, default 30
        Positive count of equal-width score bins.

    Returns
    -------
    None
        The figure is saved to disk; no table is returned.

    Raises
    ------
    ValueError
        Inputs are empty, invalid, misaligned, or a group has no positive
        weight.

    Notes
    -----
    Creates parent directories for save_fig and closes the figure after saving.
    Histograms use histogram_bins equal-width bins in [0, 1] and
    normalize weights within each group. Unlabeled rows are not confirmed
    negatives.
    """
    if histogram_bins < 1:
        raise ValueError("histogram_bins must be positive")
    y, p, w = pu_arrays(labels, probabilities, weights)
    fig, ax = plt.subplots(figsize=(9, 6))
    with saved_figure(fig, save_fig, config=figure):
        for value, label in ((1, "Known positive"), (0, "Unlabeled")):
            group = y == value
            ax.hist(
                p[group],
                bins=np.linspace(0, 1, histogram_bins + 1),
                weights=w[group] / w[group].sum(),
                alpha=0.55,
                label=label,
            )
        ax.set(
            xlabel="Model score",
            ylabel="Weighted share",
            title="PU score distribution",
            xlim=(0, 1),
        )
        ax.legend()
        fig.tight_layout()


def positive_capture(
    labels: ArrayLike,
    probabilities: ArrayLike,
    save_fig: str | Path,
    *,
    weights: ArrayLike | None = None,
    figure: FigureConfig | None = None,
) -> None:
    """Save known-positive capture against unlabeled review capacity.

    Parameters
    ----------
    labels : array-like of shape (n_samples,)
        1 identifies a known positive; 0 identifies an unlabeled observation.
    probabilities : array-like of shape (n_samples,)
        Finite model scores in [0, 1].
    save_fig : str or pathlib.Path
        Output figure path; parent directories are created.
    weights : array-like or None, default None
        Finite, nonnegative aligned weights with positive total in each group;
        None uses equal weights.

    figure : FigureConfig or None, default None
        Optional export dimensions, DPI, and format. None preserves native
        dimensions and the requested path format.

    Returns
    -------
    None
        The figure is saved to disk; no table is returned.

    Raises
    ------
    ValueError
        Inputs are empty, invalid, misaligned, or a group has no positive
        weight.

    See Also
    --------
    ml_toolkit.metrics.positive_capture_curve : Calculate the underlying fractions.

    Notes
    -----
    Creates parent directories for save_fig and closes the figure after saving.
    Cutoffs include all tied scores. Both axes show percentages; capture
    approximates recall only under representative positive labeling.
    """
    u, capture = positive_capture_curve(labels, probabilities, weights)
    fig, ax = plt.subplots(figsize=(9, 6))
    with saved_figure(fig, save_fig, config=figure):
        ax.plot(u * 100, capture * 100, label="Model")
        ax.plot(
            [0, 100], [0, 100], color="gray", linestyle="--", label="Random ranking"
        )
        ax.set(
            xlabel="Unlabeled cases flagged (%)",
            ylabel="Known positives captured (%)",
            title="Positive capture by review capacity",
            xlim=(0, 100),
            ylim=(0, 100),
        )
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
