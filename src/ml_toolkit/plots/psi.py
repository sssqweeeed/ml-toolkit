"""PSI score and time-based level-distribution figures."""

from collections.abc import Callable
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import polars as pl

from ..types import FloatArray, TimePeriod
from .._time import PERIOD_INTERVALS, period_starts
from ..config import FigureConfig
from .utils import save_figure


def feature_psi_plot(
    scores: pl.DataFrame, save_fig: Path, *, figure: FigureConfig | None = None
) -> None:
    """Save a horizontal bar chart of PSI values by feature.

    Parameters
    ----------
    scores : polars.DataFrame
        Nonempty frame with feature and finite psi columns, such as
        PSIResult.psi.
    save_fig : pathlib.Path
        Output figure path; pass a Path object, not a string.

    figure : FigureConfig or None, default None
        Optional export dimensions, DPI, and format. None preserves native
        dimensions and the requested path format.

    Returns
    -------
    None
        The figure is saved to disk; no table is returned.

    Raises
    ------
    ValueError
        Required columns are missing, scores are empty, or PSI values are
        non-finite.

    Notes
    -----
    Creates parent directories for save_fig and closes the figure after saving.
    """
    required = {"feature", "psi"}
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"PSI score columns are missing: {missing}")
    if scores.is_empty():
        raise ValueError("PSI scores must not be empty")
    ordered = scores.select("feature", "psi").sort("psi")
    names = ordered.get_column("feature").to_list()
    values = ordered.get_column("psi").to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("PSI scores must be finite")

    fig, ax = plt.subplots(figsize=(9, max(3, 0.45 * len(names) + 1)))
    ax.barh(names, values, color="tab:blue")
    ax.set(
        xlabel="Population Stability Index", ylabel="Feature", title="PSI by feature"
    )
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, save_fig, config=figure)


def time_psi_level_distribution(
    distribution: pl.DataFrame,
    feature: str,
    save_fig: Path,
    *,
    period: TimePeriod = "month",
    figure: FigureConfig | None = None,
) -> None:
    """Save time-based fitted-level shares as stacked bars at calendar-period starts.

    Parameters
    ----------
    distribution : polars.DataFrame
        Time share table with feature, period_start, sample, level, and share
        columns, such as PSIResult.time_distribution. Reference must precede
        current; only their boundary period may be shared.
    feature : str
        Feature name present in the distribution table.
    save_fig : pathlib.Path
        Output figure path; pass a Path object, not a string.

    period : TimePeriod, default "month"
        Period used to build the table; determines calendar bar widths.
    figure : FigureConfig or None, default None
        Optional export dimensions, DPI, and format. None preserves native
        dimensions and the requested path format.

    Returns
    -------
    None
        The figure is saved to disk; no table is returned.

    Raises
    ------
    ValueError
        Required columns or the requested feature are missing, or reference
        periods extend beyond the first current period.

    Notes
    -----
    Creates parent directories for save_fig and closes the figure after saving.
    Samples share one calendar axis, with a boundary at the first current bar.
    Missing periods remain empty. A shared boundary period is split into two
    equal-width bars, reference then current. This is a display convention,
    not the exact timestamp of the split. PSIAnalysis and compare_psi validate
    the original time ranges before aggregation; a manually prepared table
    must satisfy the same constraint, which period starts alone cannot verify.
    """
    required = {"feature", "period_start", "sample", "level", "share"}
    missing = sorted(required - set(distribution.columns))
    if missing:
        raise ValueError(f"PSI level distribution columns are missing: {missing}")

    feature_data = distribution.filter(pl.col("feature") == feature)
    if feature_data.is_empty():
        raise ValueError(f"Feature {feature!r} is missing from PSI distribution")

    if period not in PERIOD_INTERVALS:
        raise ValueError("Unsupported time period")
    levels = feature_data.get_column("level").unique(maintain_order=True).to_list()
    periods = (
        feature_data.select("period_start", "sample").unique()
        # Reference first, then current; chronological within each sample.
        .sort(["sample", "period_start"], descending=[True, False])
    )
    starts = periods["period_start"]
    # Matplotlib does not provide a typed date converter.
    date_numbers = cast(
        Callable[[object], FloatArray],
        mdates.date2num,  # pyright: ignore[reportUnknownMemberType]
    )
    x = date_numbers(starts.to_list())
    ends = period_starts(starts.dt.offset_by(PERIOD_INTERVALS[period]), period)
    width = date_numbers(ends.to_list()) - x
    boundary_index = periods.filter(pl.col("sample") == "reference").height
    if 0 < boundary_index < len(periods):
        previous, following = boundary_index - 1, boundary_index
        if x[previous] > x[following]:
            raise ValueError(
                "Reference periods must precede current periods; "
                "only the boundary period may be shared"
            )
        if x[previous] == x[following]:
            width[previous : following + 1] /= 2
            x[following] += width[previous]
    bottom = np.zeros(len(periods))
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.get_cmap("tab20")
    for index, level in enumerate(levels):
        shares = (
            periods.join(
                feature_data.filter(pl.col("level") == level).select(
                    "period_start", "sample", "share"
                ),
                on=["period_start", "sample"],
                how="left",
            )["share"]
            .fill_null(0.0)
            .to_numpy()
        )
        ax.bar(
            x,
            shares,
            width=width,
            align="edge",
            bottom=bottom,
            color=colors(index % 20),
            linewidth=0,
            antialiased=False,
            label=level,
        )
        bottom += shares

    if boundary_index < len(periods):
        boundary = float(x[boundary_index])
        ax.axvline(boundary, color="black", linestyle="--", linewidth=1.2)
        if boundary_index:
            ax.annotate(
                "REFERENCE",
                (float(x[0]), 0.98),
                xycoords=ax.get_xaxis_transform(),
                xytext=(4, 0),
                textcoords="offset points",
                va="top",
            )
        ax.annotate(
            "CURRENT",
            (boundary, 0.98),
            xycoords=ax.get_xaxis_transform(),
            xytext=(4, 0),
            textcoords="offset points",
            va="top",
        )
    zone = starts.to_list()[0].tzinfo
    x_axis = ax.get_xaxis()
    x_axis.axis_date(tz=zone)
    locator = mdates.AutoDateLocator(tz=zone)
    x_axis.set_major_locator(locator)
    x_axis.set_major_formatter(mdates.ConciseDateFormatter(locator, tz=zone))
    ax.set(
        title=f"Time PSI level distribution — {feature}",
        ylabel="Share of rows",
        xlabel=f"Period start ({period})",
        ylim=(0, 1),
    )
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{value:.0%}" for value in np.linspace(0, 1, 6)])
    ax.grid(axis="y", alpha=0.2)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.13),
        ncol=min(6, len(levels)),
        frameon=False,
    )
    fig.tight_layout()
    save_figure(fig, save_fig, config=figure)
