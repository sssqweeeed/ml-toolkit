"""Plots for positive–unlabeled evaluation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..metrics.pu import _inputs, positive_capture_curve
from .utils import save_figure


def score_distribution(labels, probabilities, save_fig: str | Path, *, weights=None) -> None:
    """Compare model scores for known positives and unlabeled rows."""
    y, p, w = _inputs(labels, probabilities, weights)
    fig, ax = plt.subplots(figsize=(9, 6))
    for value, label in ((1, "Known positive"), (0, "Unlabeled")):
        group = y == value
        ax.hist(p[group], bins=np.linspace(0, 1, 31), weights=w[group] / w[group].sum(),
                alpha=0.55, label=label)
    ax.set(xlabel="Model score", ylabel="Weighted share", title="PU score distribution", xlim=(0, 1))
    ax.legend()
    fig.tight_layout()
    save_figure(fig, save_fig)


def positive_capture(labels, probabilities, save_fig: str | Path, *, weights=None) -> None:
    """Plot known-positive capture against the share of U flagged."""
    u, capture = positive_capture_curve(labels, probabilities, weights)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(u * 100, capture * 100)
    ax.plot([0, 100], [0, 100], color="gray", linestyle="--", label="Random ranking")
    ax.set(xlabel="Unlabeled cases flagged (%)", ylabel="Known positives captured (%)",
           title="Positive capture by review capacity", xlim=(0, 100), ylim=(0, 100))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, save_fig)
