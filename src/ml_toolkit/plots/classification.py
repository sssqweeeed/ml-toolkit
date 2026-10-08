"""Binary classification precision, recall, F1, and calibration plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from ..config import ClassificationPlotsConfig, FigureConfig
from .utils import PlotError, numeric_vector, same_length, save_figure


def binary_curves(observed: ArrayLike, probabilities: ArrayLike) -> pd.DataFrame:
    """Compute unweighted binary scores at each distinct probability threshold.

    Parameters
    ----------
    observed : array-like of shape (n_samples,)
        Nonempty binary ground-truth labels, containing only 0 and 1.
    probabilities : array-like of shape (n_samples,)
        Finite positive-class probabilities in [0, 1], aligned with labels.

    Returns
    -------
    pandas.DataFrame
        Columns threshold, precision, recall, and f1 at distinct descending
        score thresholds; all tied scores enter together.

    Raises
    ------
    PlotError
        Inputs are empty, non-finite, misaligned, or outside the
        label/probability domain.

    Notes
    -----
    Labels are predicted with probability >= threshold. Recall and F1 are zero
    when there are no positives. Requires binary ground truth, not PU labels.

    Examples
    --------
    >>> from ml_toolkit.plots.classification import binary_curves
    >>> curve = binary_curves([0, 1], [0.2, 0.8])
    >>> curve["threshold"].tolist()
    [0.8, 0.2]
    >>> float(curve["f1"].iloc[0])
    1.0
    """
    y = numeric_vector(observed, "observed")
    p = numeric_vector(probabilities, "probabilities", probability=True)
    same_length(observed=y, probabilities=p)
    if not np.isin(y, [0.0, 1.0]).all():
        raise PlotError("observed must contain binary labels (0 or 1)")
    order = np.argsort(-p, kind="stable")
    sorted_y, sorted_p = y[order], p[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_p) != 0), len(y) - 1]
    tp = np.cumsum(sorted_y)[ends]
    predicted = ends + 1
    positives = y.sum()
    return pd.DataFrame(
        {
            "threshold": sorted_p[ends],
            "precision": tp / predicted,
            "recall": tp / positives if positives > 0 else np.zeros(len(ends)),
            "f1": (
                2 * tp / (predicted + positives)
                if positives > 0
                else np.zeros(len(ends))
            ),
        }
    )


def precision_recall(
    observed: ArrayLike,
    probabilities: ArrayLike,
    save_fig: str | Path,
    *,
    config: ClassificationPlotsConfig | None = None,
    figure: FigureConfig | None = None,
) -> pd.DataFrame:
    """Save a precision-recall curve and return its full threshold table.

    Parameters
    ----------
    observed : array-like of shape (n_samples,)
        Nonempty binary ground-truth labels, containing only 0 and 1.
    probabilities : array-like of shape (n_samples,)
        Finite positive-class probabilities in [0, 1], aligned with labels.
    save_fig : str or pathlib.Path
        Output figure path; parent directories are created.

    figure : FigureConfig or None, default None
        Optional export dimensions, DPI, and format. None preserves native
        dimensions and the requested path format.
    config : ClassificationPlotsConfig or None, default None
        Rendering point limit and bin counts. None uses defaults.

    Returns
    -------
    pandas.DataFrame
        Columns threshold, precision, recall, and f1 at distinct descending
        score thresholds; all tied scores enter together.

    Raises
    ------
    PlotError
        Binary labels or probabilities are invalid or misaligned.

    Notes
    -----
    Creates parent directories for save_fig and closes the figure after saving.
    The plot uses at most config.max_curve_points; the returned table retains every
    threshold.
    """
    config = ClassificationPlotsConfig() if config is None else config
    curve = binary_curves(observed, probabilities)
    prevalence = float(np.mean(np.asarray(observed, dtype=float)))
    ap = float(
        (
            np.diff(np.r_[0.0, curve.recall.to_numpy()]) * curve.precision.to_numpy()
        ).sum()
    )
    shown = curve.iloc[
        np.linspace(
            0, len(curve) - 1, min(len(curve), config.max_curve_points), dtype=int
        )
    ]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(
        np.r_[0.0, shown.recall],
        np.r_[1.0, shown.precision],
        label=f"Model (AP={ap:.3f})",
    )
    ax.axhline(prevalence, color="gray", linestyle="--", label="Positive prevalence")
    if prevalence > 0:
        best = curve.iloc[int(np.argmax(curve.f1.to_numpy()))]
        ax.scatter(best.recall, best.precision, color="tab:red", zorder=3)
        ax.annotate(
            f"Best F1={best.f1:.3f}\nthreshold={best.threshold:.3g}",
            (best.recall, best.precision),
            xytext=(8, -30),
            textcoords="offset points",
        )
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision–recall")
    ax.set(xlim=(0, 1), ylim=(0, 1.02))
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    save_figure(fig, save_fig, config=figure)
    return curve


def f1_threshold(
    observed: ArrayLike,
    probabilities: ArrayLike,
    save_fig: str | Path,
    *,
    config: ClassificationPlotsConfig | None = None,
    figure: FigureConfig | None = None,
) -> pd.DataFrame:
    """Save F1 against distinct score thresholds and return the curve table.

    Parameters
    ----------
    observed : array-like of shape (n_samples,)
        Nonempty binary ground-truth labels, containing only 0 and 1.
    probabilities : array-like of shape (n_samples,)
        Finite positive-class probabilities in [0, 1], aligned with labels.
    save_fig : str or pathlib.Path
        Output figure path; parent directories are created.

    figure : FigureConfig or None, default None
        Optional export dimensions, DPI, and format. None preserves native
        dimensions and the requested path format.
    config : ClassificationPlotsConfig or None, default None
        Rendering point limit and bin counts. None uses defaults.

    Returns
    -------
    pandas.DataFrame
        Columns threshold, precision, recall, and f1 at distinct descending
        score thresholds; all tied scores enter together.

    Raises
    ------
    PlotError
        Binary labels or probabilities are invalid or misaligned.

    Notes
    -----
    Creates parent directories for save_fig and closes the figure after saving.
    The plot uses at most config.max_curve_points; the returned table retains every
    threshold.
    """
    config = ClassificationPlotsConfig() if config is None else config
    curve = binary_curves(observed, probabilities)
    shown = curve.iloc[
        np.linspace(
            0, len(curve) - 1, min(len(curve), config.max_curve_points), dtype=int
        )
    ]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(shown.threshold, shown.f1)
    if np.any(np.asarray(observed) == 1):
        best = curve.iloc[int(np.argmax(curve.f1.to_numpy()))]
        ax.scatter(best.threshold, best.f1, color="tab:red", zorder=3)
        ax.axvline(best.threshold, color="tab:red", linestyle="--", alpha=0.7)
        ax.set_title(f"F1 by threshold — best={best.f1:.3f} at {best.threshold:.3g}")
    else:
        ax.set_title("F1 by threshold — no positive examples")
    ax.set(xlabel="Probability threshold", ylabel="F1", xlim=(0, 1), ylim=(0, 1.02))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_figure(fig, save_fig, config=figure)
    return curve


def calibration(
    observed: ArrayLike,
    probabilities: ArrayLike,
    save_fig: str | Path,
    *,
    config: ClassificationPlotsConfig | None = None,
    figure: FigureConfig | None = None,
) -> pd.DataFrame:
    """Save a quantile-bin calibration plot and return bin statistics.

    Parameters
    ----------
    observed : array-like of shape (n_samples,)
        Nonempty binary ground-truth labels, containing only 0 and 1.
    probabilities : array-like of shape (n_samples,)
        Finite positive-class probabilities in [0, 1], aligned with labels.
    save_fig : str or pathlib.Path
        Output figure path; parent directories are created.
    config : ClassificationPlotsConfig or None, default None
        Calibration and histogram bin counts. None uses defaults. Repeated
        score values can reduce the number of distinct calibration bins.

    figure : FigureConfig or None, default None
        Optional export dimensions, DPI, and format. None preserves native
        dimensions and the requested path format.

    Returns
    -------
    pandas.DataFrame
        Columns bin, observed_rate, predicted_rate, and count for populated
        bins; rates are unweighted means.

    Raises
    ------
    PlotError
        Binary inputs are invalid.

    Notes
    -----
    Creates parent directories for save_fig and closes the figure after saving.
    Includes a prediction histogram and a perfect-calibration reference line.
    """
    config = ClassificationPlotsConfig() if config is None else config
    n_bins = config.calibration_bins
    binary_curves(observed, probabilities)
    y = np.asarray(observed, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    bins = np.searchsorted(edges[1:-1], p, side="left")
    grouped = (
        pd.DataFrame({"bin": bins, "observed": y, "predicted": p})
        .groupby("bin", as_index=False)
        .agg(
            observed_rate=("observed", "mean"),
            predicted_rate=("predicted", "mean"),
            count=("observed", "size"),
        )
    )
    fig, (ax, histogram) = plt.subplots(
        2, 1, figsize=(9, 8), gridspec_kw={"height_ratios": [3, 1.2]}
    )
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Perfect calibration")
    ax.plot(grouped.predicted_rate, grouped.observed_rate, marker="o", label="Model")
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed rate",
        title="Probability calibration",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    histogram.hist(p, bins=config.histogram_bins, color="tab:blue")
    histogram.set(xlabel="Predicted probability", ylabel="Cases", xlim=(0, 1))
    histogram.set_yscale("log")
    fig.tight_layout()
    save_figure(fig, save_fig, config=figure)
    return grouped
