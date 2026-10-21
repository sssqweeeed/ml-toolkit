"""Time distributions of numerical features."""

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from ..types import TimePeriod
from .._time import paired_period_starts
from ..config import FigureConfig, NumericalFeatureDistributionConfig
from .utils import save_figure


def _quantile_name(quantile: float) -> str:
    return f"p{quantile * 100:02g}".replace(".", "_")


def time_feature_distribution(
    reference: pl.DataFrame,
    current: pl.DataFrame,
    feature: str,
    timestamp: str,
    save_fig: str | Path,
    *,
    period: TimePeriod = "month",
    config: NumericalFeatureDistributionConfig | None = None,
    figure: FigureConfig | None = None,
) -> pl.DataFrame:
    """Plot configured time-based quantile bands, median, and mean.

    Parameters
    ----------
    reference : polars.DataFrame
        Reference frame containing feature and timestamp columns.
    current : polars.DataFrame
        Current frame containing the same required columns. Its earliest
        timestamp must be at or after the latest reference timestamp.
        The split may fall inside one grouping period.
    feature : str
        Numerical feature to summarize.
    timestamp : str
        Date/Datetime column used for calendar grouping.
    save_fig : str or pathlib.Path
        Output figure path; parent directories are created.
    period : TimePeriod, default "month"
        Calendar grouping in the matching input timezones. Minute/hour need
        Datetime; weeks start on Monday. Empty periods are not filled.
    config : NumericalFeatureDistributionConfig or None, default None
        Quantile bands and displayed mean/median. None uses defaults. The
        features list controls workflow selection; this call plots feature.
    figure : FigureConfig or None, default None
        Export dimensions, resolution, and format; None preserves native output.

    Returns
    -------
    polars.DataFrame
        period_start, split, mean, p50, and configured endpoint quantiles named by
        percentage, such as p05 or p2_5. Defaults produce period_start, split, p05, p25,
        p50, mean, p75, p95. Summaries are unweighted.

    Raises
    ------
    ValueError
        Required columns are missing, timezones differ, or the latest reference
        timestamp exceeds the earliest current timestamp.
    TypeError
        Feature is not numerical or timestamp is not Date/Datetime.

    Notes
    -----
    Null and non-finite values are omitted. Saves and closes the figure,
    including when there are no finite values.
    """
    config = NumericalFeatureDistributionConfig() if config is None else config
    quantiles = sorted({0.5, *(q for band in config.quantile_bands for q in band)})
    columns = ["period_start", "split"]
    columns += [_quantile_name(q) for q in quantiles if q <= 0.5]
    columns += ["mean"]
    columns += [_quantile_name(q) for q in quantiles if q > 0.5]
    if timestamp not in reference.columns or timestamp not in current.columns:
        raise ValueError(f"Both samples need timestamp column {timestamp!r}")
    reference_start, current_start = paired_period_starts(
        reference[timestamp], current[timestamp], period
    )
    frames: list[pl.DataFrame] = []
    for split, data, starts in (
        ("reference", reference, reference_start),
        ("current", current, current_start),
    ):
        if feature not in data.columns or timestamp not in data.columns:
            raise ValueError(f"{split} data needs {feature!r} and {timestamp!r}")
        if not data.schema[feature].is_numeric():
            raise TypeError(f"{feature!r} must be numerical")
        if (
            data.schema[timestamp] != pl.Date
            and data.schema[timestamp].base_type() != pl.Datetime
        ):
            raise TypeError(f"{timestamp!r} must be a date or datetime")
        frame = (
            data.select(
                starts,
                pl.col(feature).cast(pl.Float64).alias("value"),
            )
            .filter(pl.col("value").is_not_null() & pl.col("value").is_finite())
            .group_by("period_start")
            .agg(
                *(
                    pl.col("value")
                    .quantile(q, interpolation="linear")
                    .alias(_quantile_name(q))
                    for q in quantiles
                ),
                pl.col("value").mean().alias("mean"),
            )
            .with_columns(pl.lit(split).alias("split"))
            .select(columns)
            .sort("period_start")
        )
        frames.append(frame)
    grouped = pl.concat(frames)
    fig, ax = plt.subplots(figsize=(14, 6))
    for split, color, linestyle in (
        ("reference", "tab:blue", "-"),
        ("current", "tab:orange", "--"),
    ):
        part = grouped.filter(pl.col("split") == split)
        if part.is_empty():
            continue
        period_start = part.get_column("period_start").to_list()
        for index, (lower, upper) in enumerate(config.quantile_bands):
            ax.fill_between(
                period_start,
                part[_quantile_name(lower)].to_numpy(),
                part[_quantile_name(upper)].to_numpy(),
                color=color,
                alpha=min(0.10 + 0.15 * index, 0.5),
                label=f"{split} {lower * 100:g}–{upper * 100:g}%",
            )
        if config.show_median:
            ax.plot(
                period_start,
                part["p50"].to_numpy(),
                color=color,
                linestyle=linestyle,
                marker="o",
                markersize=3,
                label=f"{split} median",
            )
        if config.show_mean:
            ax.plot(
                period_start,
                part["mean"].to_numpy(),
                color=color,
                linestyle=":",
                label=f"{split} mean",
            )
    current_periods = grouped.filter(pl.col("split") == "current")["period_start"]
    if not current_periods.is_empty():
        ax.axvline(
            current_periods.min(),
            color="tab:orange",
            linestyle=":",
            label="Current begins",
        )
    ax.set(
        title=f"Time distribution — {feature} ({period})",
        xlabel=f"Period start ({period})",
        ylabel=feature,
    )
    ax.grid(alpha=0.2)
    if grouped.is_empty():
        ax.text(
            0.5,
            0.5,
            "No finite values",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    elif ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    save_figure(fig, save_fig, config=figure)
    return grouped
