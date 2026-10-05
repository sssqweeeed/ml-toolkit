"""Cumulative gains and lift figures for ranking results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..metrics.ranking import RankingMetrics

def gains_curve(
    ranking: RankingMetrics,
    save_fig: Path,
) -> None:
    """Save cumulative gains and selected exposure-capacity annotations.

    Parameters
    ----------
    ranking : RankingMetrics
        Precomputed results from ml_toolkit.metrics.ranking_metrics.
    save_fig : pathlib.Path
        Output figure path; pass a Path object, not a string.

    Notes
    -----
    Creates parent directories for save_fig and closes the figure after saving.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        ranking.cumulative_exposure,
        ranking.cumulative_observed,
        label="Model",
    )
    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        label="Random",
    )

    for point in ranking.selected_points:
        share = point.percent / 100.0
        ax.scatter(
            [share],
            [point.gain],
            zorder=3,
        )
        ax.annotate(
            f"Gains@{point.percent:g}% = {point.gain:.3f}",
            xy=(share, point.gain),
            xytext=(8, 8),
            textcoords="offset points",
        )

    ax.set_title(
        "Gains curve "
        f"— Gini={ranking.gini:.3f}, "
        f"Normalized Gini={ranking.normalized_gini:.3f}"
    )
    ax.set_xlabel("Cumulative exposure")
    ax.set_ylabel("Cumulative observed")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()

    save_fig.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(save_fig, bbox_inches="tight")
    finally:
        plt.close(fig)


def lift_curve(
    ranking: RankingMetrics,
    save_fig: Path,
) -> None:
    """Save cumulative lift and selected exposure-capacity annotations.

    Parameters
    ----------
    ranking : RankingMetrics
        Precomputed results from ml_toolkit.metrics.ranking_metrics.
    save_fig : pathlib.Path
        Output figure path; pass a Path object, not a string.

    Notes
    -----
    Creates parent directories for save_fig and closes the figure after saving.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    finite_mask = np.isfinite(ranking.cumulative_lift)
    ax.plot(
        ranking.cumulative_exposure[finite_mask],
        ranking.cumulative_lift[finite_mask],
        label="Model",
    )
    ax.axhline(
        1.0,
        linestyle="--",
        label="Random",
    )

    for point in ranking.selected_points:
        share = point.percent / 100.0
        ax.scatter(
            [share],
            [point.lift],
            zorder=3,
        )
        ax.annotate(
            f"Lift@{point.percent:g}% = {point.lift:.3f}",
            xy=(share, point.lift),
            xytext=(8, 8),
            textcoords="offset points",
        )

    ax.set_title("Cumulative lift curve")
    ax.set_xlabel("Cumulative exposure")
    ax.set_ylabel("Cumulative lift")
    ax.set_xlim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()

    save_fig.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(save_fig, bbox_inches="tight")
    finally:
        plt.close(fig)
