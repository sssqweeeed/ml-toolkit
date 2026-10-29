"""Metric curves for explicit perturbations of held-out model inputs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from ..config import FigureConfig
from .utils import PlotError, save_figure


def robustness_curve(
    table: pl.DataFrame,
    *,
    metric: str,
    save_fig: str | Path,
    figure: FigureConfig | None = None,
) -> None:
    """Plot repeated scenario scores and the unmodified held-out baseline.

    Parameters
    ----------
    table : polars.DataFrame
        Long table from RobustnessCurve.calculate, containing mode, level,
        is_baseline, metric and nullable value columns.
    metric : str
        Metric name to draw in its natural units.
    save_fig : str or pathlib.Path
        Destination file. Parent directories are created.
    figure : FigureConfig or None, default None
        Shared export dimensions, resolution and file format.

    Notes
    -----
    Curves show the mean across finite repeats, with one standard deviation
    shaded. A horizontal line marks the baseline. Undefined levels leave gaps.
    The figure is closed after saving or failure.
    """
    required = {"mode", "level", "is_baseline", "metric", "value", "split"}
    if not required <= set(table.columns):
        raise PlotError("Robustness table is missing required columns")
    selected = table.filter(pl.col("metric") == metric)
    if selected.is_empty():
        raise PlotError(f"No robustness values for metric {metric!r}")
    modes = selected["mode"].unique().to_list()
    if len(modes) != 1 or modes[0] not in ("missing", "noise", "rounding", "unknown"):
        raise PlotError("Robustness plot requires one supported mode")
    mode = str(modes[0])
    labels = {
        "missing": "Fraction of rows set to missing per feature",
        "unknown": "Fraction of rows with unknown categories per feature",
        "noise": "Gaussian noise / training standard deviation",
        "rounding": "Retained decimal places (fewer is coarser)",
    }
    fig, ax = plt.subplots(figsize=(9, 5))
    try:
        for index, part in enumerate(
            selected["split"].unique(maintain_order=True).to_list()
        ):
            color = f"C{index}"
            current = selected.filter(pl.col("split") == part)
            baseline = current.filter(pl.col("is_baseline"))["value"].drop_nulls()
            summary = (
                current.filter(~pl.col("is_baseline"))
                .group_by("level")
                .agg(
                    pl.col("value").mean().alias("mean"),
                    pl.col("value").std().fill_null(0.0).alias("std"),
                )
                .sort("level")
            )
            if summary.is_empty():
                raise PlotError("Robustness table requires at least one scenario level")
            x = summary["level"].to_numpy()
            mean, deviation = summary["mean"].to_numpy(), summary["std"].to_numpy()
            ax.plot(
                x,
                mean,
                color=color,
                marker="o",
                markersize=4,
                label=f"{part.title()} repeat mean",
            )
            ax.fill_between(
                x,
                mean - deviation,
                mean + deviation,
                color=color,
                alpha=0.15,
                label="±1 repeat SD",
            )
            if len(baseline):
                value = float(baseline[0])
                if not np.isfinite(value):
                    raise PlotError("Baseline values must be finite or null")
                ax.axhline(
                    value,
                    color=color,
                    linestyle="--",
                    linewidth=1,
                    label=f"{part.title()} unmodified",
                )
        ax.set(xlabel=labels[mode], ylabel=metric, title=f"Input robustness: {mode}")
        if mode == "rounding":
            levels = (
                selected.filter(~pl.col("is_baseline"))["level"]
                .drop_nulls()
                .to_numpy()
                .astype(float)
            )
            low, high = float(levels.min()), float(levels.max())
            margin = (high - low or 1.0) * 0.05
            ax.set_xlim(high + margin, low - margin)
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        save_figure(fig, save_fig, config=figure)
    finally:
        plt.close(fig)
