"""Visualize oriented score changes from held-out feature permutations."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from ..config import FigureConfig
from .utils import PlotError, save_figure


def permutation_importance(
    table: pl.DataFrame,
    *,
    save_fig: str | Path,
    figure: FigureConfig | None = None,
) -> None:
    """Save mean importance with repeat standard deviations, in score units.

    Parameters
    ----------
    table : polars.DataFrame
        One PermutationImportance.calculate table containing one score/split.
        Null importances are omitted from the summary. Groups without finite
        repeats are shown as unavailable, never as zero importance.
    save_fig : str or pathlib.Path
        Destination image. Parent directories are created.
    figure : FigureConfig or None, default None
        Shared output dimensions, DPI and format. The figure is always closed.
    """
    required = {"group_id", "features", "metric", "split", "importance"}
    if not required <= set(table.columns) or table.is_empty():
        raise PlotError("Permutation plot requires a nonempty importance table")
    if table["metric"].n_unique() != 1 or table["split"].n_unique() != 1:
        raise PlotError("Permutation plot requires one metric and one split")
    grouped = (
        table.group_by("group_id", maintain_order=True)
        .agg(
            pl.col("features").first(),
            pl.col("importance").mean().alias("mean"),
            pl.col("importance").std(ddof=0).alias("std"),
        )
        .sort("mean", nulls_last=True)
    )
    fig, ax = plt.subplots(figsize=(9, max(4, grouped.height * 0.35 + 1.5)))
    try:
        positions = np.arange(grouped.height)
        means = grouped["mean"].to_numpy()
        labels = [" + ".join(features) for features in grouped["features"].to_list()]
        valid = np.isfinite(means)
        ax.barh(
            positions[valid],
            means[valid],
            xerr=grouped["std"].to_numpy()[valid],
            color="C0",
            alpha=0.8,
            capsize=3,
        )
        for index in np.flatnonzero(~valid):
            ax.text(0, index, " unavailable", va="center", fontsize=9)
        ax.set_yticks(positions, labels)
        ax.axvline(0, color="0.4", linewidth=1)
        ax.set(
            xlabel=f"Score deterioration ({table['metric'][0]}; mean ± repeat SD)",
            title=f"Permutation importance — {table['split'][0]}",
        )
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        save_figure(fig, save_fig, config=figure)
    finally:
        plt.close(fig)
