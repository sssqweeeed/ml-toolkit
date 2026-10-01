"""Binary classification precision, recall, F1, and calibration plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import PlotError, numeric_vector, same_length, save_figure


def binary_curves(observed, probabilities) -> pd.DataFrame:
    """Compute curves at each distinct probability threshold."""
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
    return pd.DataFrame({
        "threshold": sorted_p[ends],
        "precision": tp / predicted,
        "recall": tp / positives if positives > 0 else np.zeros(len(ends)),
        "f1": 2 * tp / (predicted + positives) if positives > 0 else np.zeros(len(ends)),
    })


def precision_recall(observed, probabilities, save_fig: str | Path) -> pd.DataFrame:
    """Save precision–recall curve and return its full threshold table."""
    curve = binary_curves(observed, probabilities)
    prevalence = float(np.mean(observed))
    ap = float(np.sum(np.diff(np.r_[0.0, curve.recall]) * curve.precision))
    shown = curve.iloc[np.linspace(0, len(curve) - 1, min(len(curve), 1000), dtype=int)]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(np.r_[0.0, shown.recall], np.r_[1.0, shown.precision], label=f"Model (AP={ap:.3f})")
    ax.axhline(prevalence, color="gray", linestyle="--", label="Positive prevalence")
    if prevalence > 0:
        best = curve.loc[curve.f1.idxmax()]
        ax.scatter(best.recall, best.precision, color="tab:red", zorder=3)
        ax.annotate(f"Best F1={best.f1:.3f}\nthreshold={best.threshold:.3g}",
                    (best.recall, best.precision), xytext=(8, -30), textcoords="offset points")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision–recall")
    ax.set(xlim=(0, 1), ylim=(0, 1.02))
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    save_figure(fig, save_fig)
    return curve


def f1_threshold(observed, probabilities, save_fig: str | Path) -> pd.DataFrame:
    """Save F1 against distinct score thresholds."""
    curve = binary_curves(observed, probabilities)
    shown = curve.iloc[np.linspace(0, len(curve) - 1, min(len(curve), 1000), dtype=int)]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(shown.threshold, shown.f1)
    if np.any(np.asarray(observed) == 1):
        best = curve.loc[curve.f1.idxmax()]
        ax.scatter(best.threshold, best.f1, color="tab:red", zorder=3)
        ax.axvline(best.threshold, color="tab:red", linestyle="--", alpha=0.7)
        ax.set_title(f"F1 by threshold — best={best.f1:.3f} at {best.threshold:.3g}")
    else:
        ax.set_title("F1 by threshold — no positive examples")
    ax.set(xlabel="Probability threshold", ylabel="F1", xlim=(0, 1), ylim=(0, 1.02))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_figure(fig, save_fig)
    return curve


def calibration(observed, probabilities, save_fig: str | Path, *, n_bins: int = 10) -> pd.DataFrame:
    """Save an equal-count calibration plot and return its bin statistics."""
    binary_curves(observed, probabilities)
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    y = np.asarray(observed, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    bins = np.searchsorted(edges[1:-1], p, side="left")
    grouped = pd.DataFrame({"bin": bins, "observed": y, "predicted": p}).groupby("bin", as_index=False).agg(
        observed_rate=("observed", "mean"), predicted_rate=("predicted", "mean"), count=("observed", "size")
    )
    fig, (ax, histogram) = plt.subplots(2, 1, figsize=(9, 8), gridspec_kw={"height_ratios": [3, 1.2]})
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Perfect calibration")
    ax.plot(grouped.predicted_rate, grouped.observed_rate, marker="o", label="Model")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed rate", title="Probability calibration",
           xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    histogram.hist(p, bins=30, color="tab:blue")
    histogram.set(xlabel="Predicted probability", ylabel="Cases", xlim=(0, 1))
    histogram.set_yscale("log")
    fig.tight_layout()
    save_figure(fig, save_fig)
    return grouped
