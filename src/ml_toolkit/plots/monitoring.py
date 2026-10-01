"""Cross-task feature monitoring plots (excluding PSI, owned by ml-psi)."""

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from .utils import save_figure


def monthly_feature_distribution(reference: pl.DataFrame, current: pl.DataFrame,
                                 feature: str, timestamp: str, save_fig: str | Path) -> pl.DataFrame:
    """Plot monthly 5/25/50/75/95 percentiles and mean for a numeric feature."""
    frames = []
    for split, data in (("reference", reference), ("current", current)):
        if feature not in data.columns or timestamp not in data.columns:
            raise ValueError(f"{split} data needs {feature!r} and {timestamp!r}")
        if not data.schema[feature].is_numeric():
            raise TypeError(f"{feature!r} must be numerical")
        if not data.schema[timestamp].is_temporal():
            raise TypeError(f"{timestamp!r} must be a date or datetime")
        frame = (data.select(
            pl.col(timestamp).dt.truncate("1mo").cast(pl.Date).alias("month"),
            pl.col(feature).cast(pl.Float64).alias("value"))
            .filter(pl.col("value").is_not_null() & pl.col("value").is_finite())
            .group_by("month").agg(
                pl.col("value").quantile(.05, interpolation="linear").alias("p05"),
                pl.col("value").quantile(.25, interpolation="linear").alias("p25"),
                pl.col("value").median().alias("p50"),
                pl.col("value").mean().alias("mean"),
                pl.col("value").quantile(.75, interpolation="linear").alias("p75"),
                pl.col("value").quantile(.95, interpolation="linear").alias("p95"))
            .with_columns(pl.lit(split).alias("split"))
            .select("month", "split", "p05", "p25", "p50", "mean", "p75", "p95")
            .sort("month"))
        frames.append(frame)
    grouped = pl.concat(frames)
    fig, ax = plt.subplots(figsize=(14, 6))
    for split, color, linestyle in (("reference", "tab:blue", "-"), ("current", "tab:orange", "--")):
        part = grouped.filter(pl.col("split") == split)
        if part.is_empty():
            continue
        month = part.get_column("month").to_list()
        ax.fill_between(month, part.get_column("p05").to_numpy(), part.get_column("p95").to_numpy(),
                        color=color, alpha=.10, label=f"{split} 5–95%")
        ax.fill_between(month, part.get_column("p25").to_numpy(), part.get_column("p75").to_numpy(),
                        color=color, alpha=.25, label=f"{split} 25–75%")
        ax.plot(month, part.get_column("p50").to_numpy(), color=color, linestyle=linestyle,
                marker="o", markersize=3, label=f"{split} median")
        ax.plot(month, part.get_column("mean").to_numpy(), color=color, linestyle=":", label=f"{split} mean")
    current_months = grouped.filter(pl.col("split") == "current").get_column("month")
    if not current_months.is_empty():
        ax.axvline(current_months.min(), color="tab:orange", linestyle=":", label="Current begins")
    ax.set(title=f"Monthly distribution — {feature}", xlabel="Month", ylabel=feature)
    ax.grid(alpha=.2)
    if grouped.is_empty():
        ax.text(.5, .5, "No finite values", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.legend(loc="best", ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    save_figure(fig, save_fig)
    return grouped
