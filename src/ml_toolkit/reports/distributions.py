"""Numerical feature distributions over recorded calendar periods."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from .._time import PERIOD_INTERVALS, paired_period_starts
from ..data import Pool
from ..feature_levels import FeatureLevelConfig
from ..feature_levels.config import validate_features
from ..figures import FigureConfig, feature_filename, saved_figure
from ..types import TimePeriod
from .base import DataAnalyzer, data_comparison_pair

__all__ = [
    "NumericalFeatureDistributions",
    "distribution_summary",
    "time_feature_distribution",
    "validate_distribution",
]


@dataclass(frozen=True, kw_only=True)
class NumericalFeatureDistributions(DataAnalyzer):
    """Compare unweighted numerical time distributions on original features.

    Parameters
    ----------
    name : str, default 'numerical_feature_distributions'
        Unique report name and artifact namespace.
    features : tuple of str or None, default None
        Numeric feature names to display. None uses model and auxiliary numeric columns;
        an empty tuple selects none. Missing or nonnumeric names raise ValueError.
    period : TimePeriod, default 'month'
        Calendar grouping in the data timezone.
    quantile_bands : tuple of pairs, default ((0.05, 0.95), (0.25, 0.75))
        Lower and upper quantiles for shaded distribution bands.
    show_mean, show_median : bool, default True
        Whether to draw each central summary.

    Notes
    -----
    Compares train with validation or test using unweighted original values,
    independent of predictions, Pool.weight, shared feature rules, and n_bins.
    Nulls and non-finite values are omitted. Saves one indexed feature
    filename in the report root.
    """

    name: str = "numerical_feature_distributions"
    features: tuple[str, ...] | None = None
    period: TimePeriod = "month"
    quantile_bands: tuple[tuple[float, float], ...] = ((0.05, 0.95), (0.25, 0.75))
    show_mean: bool = True
    show_median: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        validate_features(self.features)
        validate_distribution(self.quantile_bands)
        if self.period not in PERIOD_INTERVALS:
            raise ValueError("Unsupported time period")

    def parameters(self) -> dict[str, object]:
        return {
            "features": self.features,
            "period": self.period,
            "quantile_bands": self.quantile_bands,
            "show_mean": self.show_mean,
            "show_median": self.show_median,
        }

    def write_artifacts(
        self,
        *,
        data: Pool,
        split: pl.Series,
        artifact_dir: Path,
        figure: FigureConfig,
        features: tuple[FeatureLevelConfig, ...] | None,
        n_bins: int,
    ) -> list[Path]:
        paths: list[Path] = []
        pair = data_comparison_pair(data, split)
        if pair is None:
            return []
        _, train, current = pair
        reference = train.to_analysis_frame()
        heldout = current.to_analysis_frame()
        names = self.features
        if names is None:
            names = tuple(
                name for name, dtype in reference.schema.items() if dtype.is_numeric()
            )
        time_column = "__ml_toolkit_plot_time__"
        while time_column in reference.columns:
            time_column += "_"
        reference = reference.with_columns(train.time.alias(time_column))
        heldout = heldout.with_columns(current.time.alias(time_column))
        for index, name in enumerate(names):
            if name not in reference.columns or not reference.schema[name].is_numeric():
                raise ValueError(f"Feature {name!r} is missing or not numerical")
            path = artifact_dir / feature_filename(index, name, figure.format)
            time_feature_distribution(
                reference,
                heldout,
                name,
                time_column,
                save_fig=path,
                period=self.period,
                quantile_bands=self.quantile_bands,
                show_mean=self.show_mean,
                show_median=self.show_median,
                figure=figure,
            )
            paths.append(path)
        return paths


def _quantile_name(quantile: float) -> str:
    return f"p{quantile * 100:02g}".replace(".", "_")


def distribution_summary(
    reference: pl.DataFrame,
    current: pl.DataFrame,
    feature: str,
    timestamp: str,
    *,
    period: TimePeriod = "month",
    quantile_bands: tuple[tuple[float, float], ...] = ((0.05, 0.95), (0.25, 0.75)),
) -> pl.DataFrame:
    """Unweighted quantiles and mean of a feature per period and sample.

    Returns the table described by :func:`time_feature_distribution`.
    """
    validate_distribution(quantile_bands)
    quantiles = sorted({0.5, *(q for band in quantile_bands for q in band)})
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
    return pl.concat(frames)


def time_feature_distribution(
    reference: pl.DataFrame,
    current: pl.DataFrame,
    feature: str,
    timestamp: str,
    save_fig: str | Path,
    *,
    period: TimePeriod = "month",
    quantile_bands: tuple[tuple[float, float], ...] = ((0.05, 0.95), (0.25, 0.75)),
    show_mean: bool = True,
    show_median: bool = True,
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
    quantile_bands : tuple of pairs
        Quantile intervals shaded on the chart.
    show_mean, show_median : bool, default True
        Display the corresponding unweighted time summaries.
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
    grouped = distribution_summary(
        reference,
        current,
        feature,
        timestamp,
        period=period,
        quantile_bands=quantile_bands,
    )
    fig, ax = plt.subplots(figsize=(14, 6))
    with saved_figure(fig, save_fig, config=figure):
        for split, color, linestyle in (
            ("reference", "tab:blue", "-"),
            ("current", "tab:orange", "--"),
        ):
            part = grouped.filter(pl.col("split") == split)
            if part.is_empty():
                continue
            period_start = part.get_column("period_start").to_list()
            for index, (lower, upper) in enumerate(quantile_bands):
                ax.fill_between(
                    period_start,
                    part[_quantile_name(lower)].to_numpy(),
                    part[_quantile_name(upper)].to_numpy(),
                    color=color,
                    alpha=min(0.10 + 0.15 * index, 0.5),
                    label=f"{split} {lower * 100:g}–{upper * 100:g}%",
                )
            if show_median:
                ax.plot(
                    period_start,
                    part["p50"].to_numpy(),
                    color=color,
                    linestyle=linestyle,
                    marker="o",
                    markersize=3,
                    label=f"{split} median",
                )
            if show_mean:
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
    return grouped


def validate_distribution(bands: tuple[tuple[float, float], ...]) -> None:
    """Validate unique quantile intervals for a numerical distribution."""
    if any(not 0 <= lo < hi <= 1 for lo, hi in bands):
        raise ValueError("quantile_bands require 0 <= lower < upper <= 1")
    if len(set(bands)) != len(bands):
        raise ValueError("quantile_bands must be unique")
