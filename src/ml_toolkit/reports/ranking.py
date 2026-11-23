"""Outcome capture and lift reports on held-out predictions."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from ..data import Pool
from ..feature_levels import FeatureLevelConfig
from ..figures import FigureConfig, saved_figure
from ..metrics.ranking import RankingMetrics, ranking_metrics
from ..types import FloatArray
from .base import PredictionEvaluator, write_heldout_plot

__all__ = ["Gains", "Lift", "gains_curve", "lift_curve"]


def _gains(
    observed: FloatArray,
    prediction: FloatArray,
    *,
    weight: FloatArray,
    save_fig: Path,
    figure: FigureConfig,
) -> None:
    if not np.all(observed >= 0) or np.sum(observed * weight) <= 0:
        raise ValueError(
            "Gains/lift require nonnegative targets with positive weighted total"
        )
    gains_curve(
        ranking_metrics(observed, prediction, weight), save_fig=save_fig, figure=figure
    )


def _lift(
    observed: FloatArray,
    prediction: FloatArray,
    *,
    weight: FloatArray,
    save_fig: Path,
    figure: FigureConfig,
) -> None:
    if not np.all(observed >= 0) or np.sum(observed * weight) <= 0:
        raise ValueError(
            "Gains/lift require nonnegative targets with positive weighted total"
        )
    lift_curve(
        ranking_metrics(observed, prediction, weight), save_fig=save_fig, figure=figure
    )


@dataclass(frozen=True, kw_only=True)
class Gains(PredictionEvaluator):
    """Plot captured observed outcome against held-out exposure capacity.

    Parameters
    ----------
    name : str, default 'gains'
        Unique report name and artifact namespace.

    Notes
    -----
    Ranks validation or test rows by descending prediction. Pool weights supply
    exposure; targets must be nonnegative with positive weighted total. The
    curve shows cumulative ``target * weight`` share, including the default 10%
    capacity point. Saves ``<split>/gains.<format>``.
    """

    name: str = "gains"

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
            data, prediction, split, artifact_dir, figure, "gains", _gains
        )


@dataclass(frozen=True, kw_only=True)
class Lift(PredictionEvaluator):
    """Plot held-out outcome capture relative to uniform selection.

    Parameters
    ----------
    name : str, default 'lift'
        Unique report name and artifact namespace.

    Notes
    -----
    Lift divides captured weighted outcome share by exposure share after
    ranking by prediction. Uses validation or test rows and the default 10%
    capacity point. Targets must be nonnegative with positive weighted total.
    Saves ``<split>/lift.<format>``.
    """

    name: str = "lift"

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
            data, prediction, split, artifact_dir, figure, "lift", _lift
        )


def gains_curve(
    ranking: RankingMetrics, save_fig: Path, *, figure: FigureConfig | None = None
) -> None:
    """Save cumulative gains and selected exposure-capacity annotations.

    Parameters
    ----------
    ranking : RankingMetrics
        Precomputed results from ml_toolkit.metrics.ranking_metrics.
    save_fig : pathlib.Path
        Output figure path; pass a Path object, not a string.

    figure : FigureConfig or None, default None
        Optional export dimensions, DPI, and format. None preserves native
        dimensions and the requested path format.

    Returns
    -------
    None
        The figure is saved to disk; no table is returned.

    Notes
    -----
    Creates parent directories for save_fig and closes the figure after saving.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    with saved_figure(fig, save_fig, config=figure):
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


def lift_curve(
    ranking: RankingMetrics, save_fig: Path, *, figure: FigureConfig | None = None
) -> None:
    """Save cumulative lift and selected exposure-capacity annotations.

    Parameters
    ----------
    ranking : RankingMetrics
        Precomputed results from ml_toolkit.metrics.ranking_metrics.
    save_fig : pathlib.Path
        Output figure path; pass a Path object, not a string.

    figure : FigureConfig or None, default None
        Optional export dimensions, DPI, and format. None preserves native
        dimensions and the requested path format.

    Returns
    -------
    None
        The figure is saved to disk; no table is returned.

    Notes
    -----
    Creates parent directories for save_fig and closes the figure after saving.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    with saved_figure(fig, save_fig, config=figure):
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
