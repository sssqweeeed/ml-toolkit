"""Monthly distributions of numerical features."""

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from ..config import FigureConfig, NumericalFeatureDistributionConfig
from .utils import save_figure


def _quantile_name(quantile: float) -> str:
    return f"p{quantile * 100:02g}".replace(".", "_")


def monthly_feature_distribution(
    reference: pl.DataFrame,
    current: pl.DataFrame,
    feature: str,
    timestamp: str,
    save_fig: str | Path,
    *,
    config: NumericalFeatureDistributionConfig | None = None,
    figure: FigureConfig | None = None,
) -> pl.DataFrame:
    """Plot configured monthly quantile bands, median, and mean.

    Parameters
    ----------
    reference : polars.DataFrame
        Reference frame containing feature and timestamp columns.
    current : polars.DataFrame
        Current frame containing the same required columns.
    feature : str
        Numerical feature to summarize.
    timestamp : str
        Date/Datetime column used for calendar-month grouping.
    save_fig : str or pathlib.Path
        Output figure path; parent directories are created.
    config : NumericalFeatureDistributionConfig or None, default None
        Quantile bands and displayed mean/median. None uses defaults. The
        features list controls workflow selection; this call plots feature.
    figure : FigureConfig or None, default None
        Export dimensions, resolution, and format; None preserves native output.

    Returns
    -------
    polars.DataFrame
        month, split, mean, p50, and configured endpoint quantiles named by
        percentage, such as p05 or p2_5. Defaults produce month, split, p05, p25,
        p50, mean, p75, p95. Summaries are unweighted.

    Raises
    ------
    ValueError
        Required columns are missing.
    TypeError
        Feature is not numerical or timestamp is not Date/Datetime.

    Notes
    -----
    Null and non-finite values are omitted. Saves and closes the figure,
    including when there are no finite values.
    """
    config = NumericalFeatureDistributionConfig() if config is None else config
    quantiles = sorted({0.5, *(q for band in config.quantile_bands for q in band)})
    columns = ["month", "split"]
    columns += [_quantile_name(q) for q in quantiles if q <= 0.5]
    columns += ["mean"]
    columns += [_quantile_name(q) for q in quantiles if q > 0.5]
    frames: list[pl.DataFrame] = []
    for split, data in (("reference", reference), ("current", current)):
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
                pl.col(timestamp).dt.truncate("1mo").cast(pl.Date).alias("month"),
                pl.col(feature).cast(pl.Float64).alias("value"),
            )
            .filter(pl.col("value").is_not_null() & pl.col("value").is_finite())
            .group_by("month")
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
            .sort("month")
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
        month = part.get_column("month").to_list()
        for index, (lower, upper) in enumerate(config.quantile_bands):
            ax.fill_between(
                month,
                part[_quantile_name(lower)].to_numpy(),
                part[_quantile_name(upper)].to_numpy(),
                color=color,
                alpha=min(0.10 + 0.15 * index, 0.5),
                label=f"{split} {lower * 100:g}–{upper * 100:g}%",
            )
        if config.show_median:
            ax.plot(
                month,
                part["p50"].to_numpy(),
                color=color,
                linestyle=linestyle,
                marker="o",
                markersize=3,
                label=f"{split} median",
            )
        if config.show_mean:
            ax.plot(
                month,
                part["mean"].to_numpy(),
                color=color,
                linestyle=":",
                label=f"{split} mean",
            )
    current_months = grouped.filter(pl.col("split") == "current")["month"]
    if not current_months.is_empty():
        ax.axvline(
            current_months.min(),
            color="tab:orange",
            linestyle=":",
            label="Current begins",
        )
    ax.set(title=f"Monthly distribution — {feature}", xlabel="Month", ylabel=feature)
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
