"""Binary-classification probability and threshold reports."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from numpy.typing import ArrayLike

from ..data import Pool
from ..feature_levels import FeatureLevelConfig
from ..figures import (
    FigureConfig,
    PlotError,
    numeric_vector,
    same_length,
    saved_figure,
    weights,
)
from ..metrics.classification import average_precision, binary_threshold_scores
from ..types import FloatArray
from .base import PredictionEvaluator, write_heldout_plot

__all__ = [
    "Calibration",
    "CalibrationBins",
    "F1Threshold",
    "PrecisionRecall",
    "PrecisionRecallSummary",
    "binary_curves",
    "calibration",
    "calibration_bins",
    "f1_threshold",
    "precision_recall",
    "precision_recall_summary",
]


@dataclass(frozen=True, kw_only=True)
class PrecisionRecall(PredictionEvaluator):
    """Plot weighted precision against recall on held-out probabilities.

    Parameters
    ----------
    name : str, default 'precision_recall'
        Unique report name and artifact namespace.
    max_curve_points : int, default 1000
        Maximum displayed threshold points; does not change metric calculations.

    Notes
    -----
    Uses the single validation or test sample, ignoring train rows. Targets are
    confirmed binary labels and predictions are probabilities in [0, 1].
    Saves ``<split>/precision_recall.<format>``. Row weights affect both curves.
    """

    name: str = "precision_recall"
    max_curve_points: int = 1000

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.max_curve_points < 1:
            raise ValueError("max_curve_points must be positive")

    def parameters(self) -> dict[str, object]:
        return {"max_curve_points": self.max_curve_points}

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
            "precision_recall",
            precision_recall,
            max_curve_points=self.max_curve_points,
        )


@dataclass(frozen=True, kw_only=True)
class F1Threshold(PredictionEvaluator):
    """Plot weighted F1 against held-out probability thresholds.

    Parameters
    ----------
    name : str, default 'f1_threshold'
        Unique report name and artifact namespace.
    max_curve_points : int, default 1000
        Maximum displayed threshold points.

    Notes
    -----
    Uses confirmed binary validation or test labels and probabilities in [0, 1].
    Saves ``<split>/f1_threshold.<format>`` without changing the model threshold.
    """

    name: str = "f1_threshold"
    max_curve_points: int = 1000

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.max_curve_points < 1:
            raise ValueError("max_curve_points must be positive")

    def parameters(self) -> dict[str, object]:
        return {"max_curve_points": self.max_curve_points}

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
            "f1_threshold",
            f1_threshold,
            max_curve_points=self.max_curve_points,
        )


@dataclass(frozen=True, kw_only=True)
class Calibration(PredictionEvaluator):
    """Compare held-out probabilities with observed positive rates.

    Parameters
    ----------
    name : str, default 'calibration'
        Unique report name and artifact namespace.
    n_bins : int, default 10
        Number of probability quantiles for the calibration curve. Independent
        of shared feature-level binning.
    histogram_bins : int, default 30
        Number of bins in the accompanying probability histogram.

    Notes
    -----
    Uses confirmed binary labels and probabilities in [0, 1]. Means and the
    histogram use weights; quantile boundaries use positive-weight rows without
    weighting. Saves ``<split>/calibration.<format>`` for validation or test.
    """

    name: str = "calibration"
    n_bins: int = 10
    histogram_bins: int = 30

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_bins < 1:
            raise ValueError("n_bins must be positive")
        if self.histogram_bins < 1:
            raise ValueError("histogram_bins must be positive")

    def parameters(self) -> dict[str, object]:
        return {"n_bins": self.n_bins, "histogram_bins": self.histogram_bins}

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
            "calibration",
            calibration,
            n_bins=self.n_bins,
            histogram_bins=self.histogram_bins,
        )


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
    This small calculation shows the precision/recall tradeoff: lowering the
    threshold from 0.9 to 0.6 finds the second positive and one false positive.

    >>> from ml_toolkit.reports.classification import binary_curves
    >>> curve = binary_curves([1, 0, 1, 0], [0.9, 0.8, 0.6, 0.2])
    >>> print(curve.iloc[[0, 2]].round(3).to_string(index=False))
     threshold  precision  recall    f1
           0.9      1.000     0.5 0.667
           0.6      0.667     1.0 0.800
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


@dataclass(frozen=True)
class PrecisionRecallSummary:
    """Threshold table with the weighted prevalence and average precision."""

    curve: pd.DataFrame
    prevalence: float
    average_precision: float


def precision_recall_summary(
    observed: ArrayLike, probabilities: ArrayLike, *, weight: ArrayLike | None = None
) -> PrecisionRecallSummary:
    """Compute the threshold table, weighted prevalence and average precision."""
    curve = binary_curves(observed, probabilities, weight=weight)
    labels = np.asarray(observed, dtype=float)
    exposure = weights(weight, len(labels))
    prevalence = float(np.average(labels, weights=exposure))
    has_positive_weight = float(np.dot(labels, exposure)) > 0
    ap = (
        average_precision(labels, np.asarray(probabilities, dtype=float), exposure)
        if has_positive_weight
        else 0.0
    )
    return PrecisionRecallSummary(curve, prevalence, ap)


def precision_recall(
    observed: ArrayLike,
    probabilities: ArrayLike,
    save_fig: str | Path,
    *,
    weight: ArrayLike | None = None,
    max_curve_points: int = 1000,
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
    max_curve_points : int, default 1000
        Positive rendering limit; calculations retain all thresholds.
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
    The plot uses at most max_curve_points; the returned table retains
    every threshold. Scores use the supplied weights.
    """
    if max_curve_points < 1:
        raise ValueError("max_curve_points must be positive")
    summary = precision_recall_summary(observed, probabilities, weight=weight)
    curve = summary.curve
    shown = curve.iloc[
        np.linspace(0, len(curve) - 1, min(len(curve), max_curve_points), dtype=int)
    ]
    fig, ax = plt.subplots(figsize=(9, 6))
    with saved_figure(fig, save_fig, config=figure):
        ax.plot(
            np.r_[0.0, shown.recall],
            np.r_[1.0, shown.precision],
            label=f"Model (weighted AP={summary.average_precision:.3f})",
        )
        ax.axhline(
            summary.prevalence,
            color="gray",
            linestyle="--",
            label="Weighted positive prevalence",
        )
        if summary.prevalence > 0:
            best = curve.iloc[int(np.argmax(curve.f1.to_numpy()))]
            ax.scatter(best.recall, best.precision, color="tab:red", zorder=3)
            ax.annotate(
                f"Best F1={best.f1:.3f}\nthreshold={best.threshold:.3g}",
                (best.recall, best.precision),
                xytext=(8, -30),
                textcoords="offset points",
            )
        ax.set(
            xlabel="Weighted recall",
            ylabel="Weighted precision",
            title="Precision–recall",
        )
        ax.set(xlim=(0, 1), ylim=(0, 1.02))
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
    return curve


def f1_threshold(
    observed: ArrayLike,
    probabilities: ArrayLike,
    save_fig: str | Path,
    *,
    weight: ArrayLike | None = None,
    max_curve_points: int = 1000,
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
    max_curve_points : int, default 1000
        Positive rendering limit; calculations retain all thresholds.
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
    The plot uses at most max_curve_points; the returned table retains
    every threshold. Scores use the supplied weights.
    """
    if max_curve_points < 1:
        raise ValueError("max_curve_points must be positive")
    curve = binary_curves(observed, probabilities, weight=weight)
    shown = curve.iloc[
        np.linspace(0, len(curve) - 1, min(len(curve), max_curve_points), dtype=int)
    ]
    fig, ax = plt.subplots(figsize=(9, 6))
    with saved_figure(fig, save_fig, config=figure):
        ax.plot(shown.threshold, shown.f1)
        if np.any(
            (np.asarray(observed) == 1)
            & (weights(weight, len(np.asarray(observed))) > 0)
        ):
            best = curve.iloc[int(np.argmax(curve.f1.to_numpy()))]
            ax.scatter(best.threshold, best.f1, color="tab:red", zorder=3)
            ax.axvline(best.threshold, color="tab:red", linestyle="--", alpha=0.7)
            ax.set_title(
                f"F1 by threshold — best={best.f1:.3f} at {best.threshold:.3g}"
            )
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
    return curve


@dataclass(frozen=True)
class CalibrationBins:
    """Quantile-bin statistics with the positive-weight rows they came from.

    Attributes
    ----------
    table : pandas.DataFrame
        Columns bin, observed_rate, predicted_rate, count and weight_sum.
    probabilities, weight : FloatArray
        Positive-weight probabilities and their weights, in input order.
    """

    table: pd.DataFrame
    probabilities: FloatArray
    weight: FloatArray


def calibration_bins(
    observed: ArrayLike,
    probabilities: ArrayLike,
    n_bins: int,
    *,
    weight: ArrayLike | None = None,
) -> CalibrationBins:
    """Group positive-weight rows into probability quantile bins."""
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
    return CalibrationBins(grouped, p, w)


def calibration(
    observed: ArrayLike,
    probabilities: ArrayLike,
    save_fig: str | Path,
    *,
    weight: ArrayLike | None = None,
    n_bins: int = 10,
    histogram_bins: int = 30,
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
    n_bins, histogram_bins : int, default 10, 30
        Positive calibration and histogram bin counts.
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
    if n_bins < 1 or histogram_bins < 1:
        raise ValueError("bin counts must be positive")
    binned = calibration_bins(observed, probabilities, n_bins, weight=weight)
    grouped, p, w = binned.table, binned.probabilities, binned.weight
    fig, (ax, histogram) = plt.subplots(
        2, 1, figsize=(9, 8), gridspec_kw={"height_ratios": [3, 1.2]}
    )
    with saved_figure(fig, save_fig, config=figure):
        ax.plot(
            [0, 1], [0, 1], color="gray", linestyle="--", label="Perfect calibration"
        )
        ax.plot(
            grouped.predicted_rate, grouped.observed_rate, marker="o", label="Model"
        )
        ax.set(
            xlabel="Weighted mean predicted probability",
            ylabel="Weighted observed rate",
            title="Probability calibration",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        histogram.hist(p, weights=w, bins=histogram_bins, color="tab:blue")
        histogram.set(
            xlabel="Predicted probability", ylabel="Sample weight", xlim=(0, 1)
        )
        histogram.set_yscale("log")
        fig.tight_layout()
    return grouped
