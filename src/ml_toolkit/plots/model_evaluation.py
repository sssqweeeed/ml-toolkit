"""Plots for metric tables calculated from fitted model variants."""

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from ..config import FigureConfig
from .utils import PlotError, save_figure


def shrink_curve(
    table: pl.DataFrame,
    *,
    metric: str,
    save_fig: str | Path,
    figure: FigureConfig | None = None,
) -> None:
    """Save one metric against retained iterations, with a line per split.

    table is the long table returned by ShrinkCurve.calculate. Null metric
    values leave gaps. Uses natural metric units without assuming a preferred
    direction. The destination's parents are created and figures always close.
    """
    required = {"n_iterations", "split", "metric", "value"}
    if not required <= set(table.columns):
        raise PlotError(
            "Shrink curve table requires n_iterations, split, metric, value"
        )
    selected = table.filter(pl.col("metric") == metric)
    if selected.is_empty():
        raise PlotError(f"No shrink curve values for metric {metric!r}")
    fig, ax = plt.subplots(figsize=(9, 5))
    try:
        for part, label in (
            ("train", "Train"),
            ("val", "Validation"),
            ("test", "Test"),
        ):
            values = selected.filter(pl.col("split") == part).sort("n_iterations")
            if not values.is_empty():
                ax.plot(
                    values["n_iterations"].to_numpy(),
                    values["value"].to_numpy(),
                    marker="o",
                    markersize=3,
                    label=label,
                )
        ax.set(
            xlabel="Retained boosting iterations",
            ylabel=metric,
            title=f"Model shrink: {metric}",
        )
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        save_figure(fig, save_fig, config=figure)
    finally:
        plt.close(fig)
