"""Weighted regression residual distribution and bias diagnostics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from .._residuals import ResidualLoss, ResidualResult
from ..config import FigureConfig
from .utils import PlotError, save_figure


def residual_diagnostics(
    result: ResidualResult,
    *,
    save_fig: str | Path,
    loss: ResidualLoss = "squared",
    histogram_bins: int = 40,
    max_samples: int = 10_000,
    seed: int = 0,
    figure: FigureConfig | None = None,
) -> None:
    """Plot weighted residual distribution, sampled points, and bin quantiles.

    Parameters
    ----------
    result : ResidualResult
        Complete tables returned by Residuals.calculate. Only positive-weight
        rows enter plots. Every supplied split appears in each panel.
    save_fig : str or pathlib.Path
        Destination figure path. Parents are created and the figure is closed.
    loss : {'squared', 'absolute', 'poisson', 'gamma'}, default 'squared'
        Residual label matching the calculation. Poisson/Gamma display signed
        square-root unit deviance; other losses use prediction minus target.
    histogram_bins : int, default 40
        Number of shared bins. Histogram heights sum to one within each split,
        using row weights; all positive-weight rows contribute.
    max_samples : int, default 10000
        Maximum scatter points per split, uniformly sampled without replacement.
        Marker size represents relative row weight. Summaries use every row.
    seed : int, default 0
        Nonnegative random seed for display sampling only.
    figure : FigureConfig or None, default None
        Shared figure size, resolution, and output format.

    Notes
    -----
    The final panel shows the weighted median, interquartile band and weighted
    mean residual at each populated prediction bin's weighted mean prediction.
    Positive residuals indicate overprediction for every supported loss.
    """
    if histogram_bins < 1 or max_samples < 1 or seed < 0:
        raise PlotError(
            "histogram_bins/max_samples must be positive and seed nonnegative"
        )
    rows = result.rows.filter(pl.col("weight") > 0)
    if rows.is_empty():
        raise PlotError("Residual plots require positive-weight observations")
    edges = np.histogram_bin_edges(rows["residual"].to_numpy(), bins=histogram_bins)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    try:
        rng = np.random.default_rng(seed)
        for part, label in (
            ("train", "Train"),
            ("val", "Validation"),
            ("test", "Test"),
        ):
            current = rows.filter(pl.col("split") == part)
            if current.is_empty():
                continue
            weight = current["weight"].to_numpy()
            axes[0].hist(
                current["residual"].to_numpy(),
                bins=edges,
                weights=weight / weight.sum(),
                histtype="step",
                linewidth=1.8,
                label=label,
            )
            if len(current) > max_samples:
                index = np.sort(rng.choice(len(current), max_samples, replace=False))
                sampled = current[index]
            else:
                sampled = current
            sizes = 8 + 22 * np.sqrt(sampled["weight"].to_numpy() / weight.max())
            axes[1].scatter(
                sampled["prediction"].to_numpy(),
                sampled["residual"].to_numpy(),
                s=sizes,
                alpha=0.25,
                edgecolors="none",
                label=label,
            )
            bins = result.bins.filter(pl.col("split") == part).sort("prediction_bin")
            x = bins["mean_prediction"].to_numpy()
            line = axes[2].plot(
                x,
                bins["p50"].to_numpy(),
                marker="o",
                markersize=3,
                label=f"{label} median",
            )[0]
            axes[2].fill_between(
                x,
                bins["p25"].to_numpy(),
                bins["p75"].to_numpy(),
                color=line.get_color(),
                alpha=0.12,
            )
            axes[2].plot(
                x,
                bins["mean_residual"].to_numpy(),
                color=line.get_color(),
                linestyle="--",
                label=f"{label} mean",
            )
        residual_label = (
            "Residual (prediction − observed)"
            if loss in ("squared", "absolute")
            else f"Signed {loss.title()} deviance residual"
        )
        axes[0].set(
            xlabel=residual_label,
            ylabel="Share of total row weight",
            title="Residual distribution",
        )
        axes[1].set(
            xlabel="Prediction",
            ylabel=residual_label,
            title="Residual versus prediction",
        )
        axes[2].set(
            xlabel="Weighted mean prediction per bin",
            ylabel=residual_label,
            title="Weighted bias and interquartile range",
        )
        axes[0].axvline(0, color="0.4", linewidth=0.8)
        for axis in axes[1:]:
            axis.axhline(0, color="0.4", linewidth=0.8)
        for axis in axes:
            axis.grid(alpha=0.15)
            axis.legend(fontsize="small")
        fig.tight_layout()
        save_figure(fig, save_fig, config=figure)
    finally:
        plt.close(fig)
