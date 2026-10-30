"""Time-based metric curves with their supporting window weights."""

from datetime import datetime
from pathlib import Path
from typing import cast

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import polars as pl

from .._time import PERIOD_INTERVALS
from ..config import FigureConfig
from ..types import FloatArray, IntArray, TimePeriod
from .utils import PlotError, save_figure


def metric_trend(
    table: pl.DataFrame,
    *,
    metric: str,
    save_fig: str | Path,
    period: TimePeriod = "month",
    figure: FigureConfig | None = None,
) -> None:
    """Plot one time-window metric above its total observation weight.

    Parameters
    ----------
    table : polars.DataFrame
        Long table returned by MetricTrend.calculate(), including period_start,
        split, metric, value, and sum_weight. Values may be null.
    metric : str
        Score name to display in its natural units.
    save_fig : str or pathlib.Path
        Output path. Parent directories are created.
    period : TimePeriod, default 'month'
        Calendar bucket used when calculating the table. Missing buckets break
        the displayed line; metric values are never interpolated.
    figure : FigureConfig or None, default None
        Shared dimensions, resolution, and output format.

    Notes
    -----
    Separate lines represent train, validation, or test. The lower panel shows
    each window's sum_weight, including zero and undersupported windows. All
    figures close after saving or failure. Calendar timestamps retain their
    timezone for axis formatting.
    """
    required = {"period_start", "split", "metric", "value", "sum_weight"}
    if not required <= set(table.columns):
        raise PlotError(
            "Metric trend table requires period_start, split, metric, value, sum_weight"
        )
    if period not in PERIOD_INTERVALS:
        raise PlotError("Unsupported time period")
    selected = table.filter(pl.col("metric") == metric)
    if selected.is_empty():
        raise PlotError(f"No metric trend values for metric {metric!r}")
    fig, (score_ax, volume_ax) = plt.subplots(
        2,
        1,
        figsize=(10, 6),
        sharex=True,
        gridspec_kw={"height_ratios": (3, 1)},
    )
    try:
        zone = None
        for index, (part, label) in enumerate(
            (("train", "Train"), ("val", "Validation"), ("test", "Test"))
        ):
            values = selected.filter(pl.col("split") == part).sort("period_start")
            if values.is_empty():
                continue
            first = cast(datetime, values["period_start"][0])
            observed = [
                cast(datetime, value) for value in values["period_start"].to_list()
            ]
            following = values["period_start"].dt.offset_by(PERIOD_INTERVALS[period])
            instants: IntArray = values["period_start"].cast(pl.Int64).to_numpy()
            next_instants: IntArray = following.cast(pl.Int64).to_numpy()
            scores: FloatArray = values["value"].to_numpy()
            volumes: FloatArray = values["sum_weight"].to_numpy()
            dates: list[datetime] = []
            shown_scores: list[float] = []
            shown_volumes: list[float] = []
            for position, timestamp in enumerate(observed):
                if position and instants[position] != next_instants[position - 1]:
                    # One separator breaks a gap of any length without allocating
                    # every absent minute/hour between sparse observations.
                    dates.append(cast(datetime, following[position - 1]))
                    shown_scores.append(float("nan"))
                    shown_volumes.append(float("nan"))
                dates.append(timestamp)
                shown_scores.append(float(scores[position]))
                shown_volumes.append(float(volumes[position]))
            zone = first.tzinfo
            color = f"C{index}"
            score_ax.plot(
                dates,
                shown_scores,
                color=color,
                marker="o",
                markersize=3,
                label=label,
            )
            volume_ax.plot(
                dates,
                shown_volumes,
                color=color,
                marker="o",
                markersize=3,
            )
        locator = mdates.AutoDateLocator(tz=zone)
        volume_ax.xaxis.set_major_locator(locator)
        volume_ax.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(locator, tz=zone)
        )
        score_ax.set(ylabel=metric, title=f"Metric trend: {metric}")
        volume_ax.set(ylabel="Window weight", xlabel="Window endpoint (period start)")
        score_ax.legend()
        score_ax.grid(alpha=0.2)
        volume_ax.grid(alpha=0.2)
        fig.tight_layout()
        save_figure(fig, save_fig, config=figure)
    finally:
        plt.close(fig)
