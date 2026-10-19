"""Binary classification precision, recall, F1, and calibration plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from ..metrics._binary import binary_threshold_scores
from ..config import ClassificationPlotsConfig, FigureConfig
from .utils import PlotError, numeric_vector, same_length, save_figure, weights


def binary_curves(
    observed: ArrayLike, probabilities: ArrayLike, *, weight: ArrayLike | None = None
) -> pd.DataFrame:
    """Compute weighted binary scores at each distinct probability threshold.

    Parameters
    ----------
    observed : array-like of shape (n_samples,)
        Nonempty binary ground-truth labels, containing only 0 and 1.
    probabilities : array-like of shape (n_samples,)
        Finite positive-class probabilities in [0, 1], aligned with labels.

    weight : array-like of shape (n_samples,) or None, default None
        Nonnegative finite sample weights with positive total; None uses ones.
        Zero-weight rows do not influence the calculations.

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
    w = weights(weight, len(y))
    try:
        threshold, precision, recall, f1 = binary_threshold_scores(y, p, w)
    except ValueError as error:
        raise PlotError(str(error)) from error
    return pd.DataFrame(
        {"threshold": threshold, "precision": precision, "recall": recall, "f1": f1}
    )


def precision_recall(
    observed: ArrayLike,
    probabilities: ArrayLike,
    save_fig: str | Path,
    *,
    weight: ArrayLike | None = None,
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
        Uses max_curve_points to limit rendered points. Other settings are
        ignored by this function. None uses defaults.

    weight : array-like of shape (n_samples,) or None, default None
        Nonnegative finite sample weights with positive total; None uses ones.
        Zero-weight rows do not influence the calculations.

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
    The plot uses at most config.max_curve_points; the returned table retains
    every threshold. Scores use the supplied weights.
    """
    config = ClassificationPlotsConfig() if config is None else config
    curve = binary_curves(observed, probabilities, weight=weight)
    prevalence = float(
        np.average(
            np.asarray(observed, dtype=float),
            weights=weights(weight, len(np.asarray(observed))),
        )
    )
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
        label=f"Model (weighted AP={ap:.3f})",
    )
    ax.axhline(
        prevalence, color="gray", linestyle="--", label="Weighted positive prevalence"
    )
    if prevalence > 0:
        best = curve.iloc[int(np.argmax(curve.f1.to_numpy()))]
        ax.scatter(best.recall, best.precision, color="tab:red", zorder=3)
        ax.annotate(
            f"Best F1={best.f1:.3f}\nthreshold={best.threshold:.3g}",
            (best.recall, best.precision),
            xytext=(8, -30),
            textcoords="offset points",
        )
    ax.set(
        xlabel="Weighted recall", ylabel="Weighted precision", title="Precision–recall"
    )
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
    weight: ArrayLike | None = None,
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
        Uses max_curve_points to limit rendered points. Other settings are
        ignored by this function. None uses defaults.

    weight : array-like of shape (n_samples,) or None, default None
        Nonnegative finite sample weights with positive total; None uses ones.
        Zero-weight rows do not influence the calculations.

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
    The plot uses at most config.max_curve_points; the returned table retains
    every threshold. Scores use the supplied weights.
    """
    config = ClassificationPlotsConfig() if config is None else config
    curve = binary_curves(observed, probabilities, weight=weight)
    shown = curve.iloc[
        np.linspace(
            0, len(curve) - 1, min(len(curve), config.max_curve_points), dtype=int
        )
    ]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(shown.threshold, shown.f1)
    if np.any(
        (np.asarray(observed) == 1) & (weights(weight, len(np.asarray(observed))) > 0)
    ):
        best = curve.iloc[int(np.argmax(curve.f1.to_numpy()))]
        ax.scatter(best.threshold, best.f1, color="tab:red", zorder=3)
        ax.axvline(best.threshold, color="tab:red", linestyle="--", alpha=0.7)
        ax.set_title(f"F1 by threshold — best={best.f1:.3f} at {best.threshold:.3g}")
    else:
        ax.set_title("F1 by threshold — no positive examples")
    ax.set(
        xlabel="Probability threshold",
        ylabel="Weighted F1",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_figure(fig, save_fig, config=figure)
    return curve


def calibration(
    observed: ArrayLike,
    probabilities: ArrayLike,
    save_fig: str | Path,
    *,
    weight: ArrayLike | None = None,
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

    weight : array-like of shape (n_samples,) or None, default None
        Nonnegative finite sample weights with positive total; None uses ones.
        Zero-weight rows do not influence the calculations.

    Returns
    -------
    pandas.DataFrame
        Columns bin, observed_rate, predicted_rate, count, and weight_sum for
        populated positive-weight bins. Rates are weighted means; quantile
        boundaries use the positive-weight observations without weighting.

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
    binary_curves(observed, probabilities, weight=weight)
    y = np.asarray(observed, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    w = weights(weight, len(y))
    keep = w > 0
    y, p, w = y[keep], p[keep], w[keep]
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    bins = np.searchsorted(edges[1:-1], p, side="left")
    grouped = (
        pd.DataFrame({"bin": bins, "observed": y * w, "predicted": p * w, "weight": w})
        .groupby("bin", as_index=False)
        .agg(
            observed_rate=("observed", "sum"),
            predicted_rate=("predicted", "sum"),
            count=("observed", "size"),
            weight_sum=("weight", "sum"),
        )
    )
    grouped["observed_rate"] /= grouped["weight_sum"]
    grouped["predicted_rate"] /= grouped["weight_sum"]
    fig, (ax, histogram) = plt.subplots(
        2, 1, figsize=(9, 8), gridspec_kw={"height_ratios": [3, 1.2]}
    )
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Perfect calibration")
    ax.plot(grouped.predicted_rate, grouped.observed_rate, marker="o", label="Model")
    ax.set(
        xlabel="Weighted mean predicted probability",
        ylabel="Weighted observed rate",
        title="Probability calibration",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    histogram.hist(p, weights=w, bins=config.histogram_bins, color="tab:blue")
    histogram.set(xlabel="Predicted probability", ylabel="Sample weight", xlim=(0, 1))
    histogram.set_yscale("log")
    fig.tight_layout()
    save_figure(fig, save_fig, config=figure)
    return grouped
