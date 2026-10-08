"""Plots for positive–unlabeled evaluation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from ..config import FigureConfig, PUPlotsConfig
from ..metrics.pu import positive_capture_curve, pu_arrays
from .utils import save_figure


def score_distribution(
    labels: ArrayLike,
    probabilities: ArrayLike,
    save_fig: str | Path,
    *,
    weights: ArrayLike | None = None,
    config: PUPlotsConfig | None = None,
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
    config : PUPlotsConfig or None, default None
        Histogram bin count. None uses PUPlotsConfig defaults.

    Raises
    ------
    ValueError
        Inputs are empty, invalid, misaligned, or a group has no positive
        weight.

    Notes
    -----
    Creates parent directories for save_fig and closes the figure after saving.
    Histograms use config.histogram_bins equal-width bins in [0, 1] and normalize weights within
    each group. Unlabeled rows are not confirmed negatives.
    """
    config = PUPlotsConfig() if config is None else config
    y, p, w = pu_arrays(labels, probabilities, weights)
    fig, ax = plt.subplots(figsize=(9, 6))
    for value, label in ((1, "Known positive"), (0, "Unlabeled")):
        group = y == value
        ax.hist(
            p[group],
            bins=np.linspace(0, 1, config.histogram_bins + 1),
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
    save_figure(fig, save_fig, config=figure)


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
    ax.plot(u * 100, capture * 100)
    ax.plot([0, 100], [0, 100], color="gray", linestyle="--", label="Random ranking")
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
    save_figure(fig, save_fig, config=figure)
