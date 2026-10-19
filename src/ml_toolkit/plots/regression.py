"""Observed-versus-fitted diagnostics for continuous targets."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from numpy.typing import ArrayLike

from ..types import TimePeriod
from .._time import paired_period_starts
from ..config import FigureConfig
from ..types import FloatArray
from .utils import PlotError, numeric_vector, same_length, save_figure, weights


def _aggregate(
    groups: pd.Series[str],
    observed: FloatArray,
    fitted: FloatArray,
    weight: FloatArray,
    name: str,
    *,
    sort: bool,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            name: groups,
            "observed_weighted": observed * weight,
            "fitted_weighted": fitted * weight,
            "volume": weight,
        }
    )
    grouped = frame.groupby(name, sort=sort, as_index=False, observed=True).sum(
        numeric_only=True
    )
    grouped = grouped.loc[grouped.volume > 0].copy()
    grouped["observed"] = grouped.observed_weighted / grouped.volume
    grouped["fitted"] = grouped.fitted_weighted / grouped.volume
    return grouped[[name, "observed", "fitted", "volume"]]


def observed_fitted(
    feature: ArrayLike,
    observed: ArrayLike,
    fitted: ArrayLike,
    save_fig: str | Path,
    *,
    weight: ArrayLike | None = None,
    feature_name: str = "Feature",
    n_bins: int = 10,
    figure: FigureConfig | None = None,
) -> pd.DataFrame:
    """Plot weighted observed and fitted means by feature group.

    Parameters
    ----------
    feature : array-like of shape (n_samples,)
        Feature values. Varying numeric features are quantile-binned;
        categorical and constant features are grouped by value. Missing values
        form a missing group.
    observed : array-like of shape (n_samples,)
        Nonempty finite observed values.
    fitted : array-like of shape (n_samples,)
        Finite predictions aligned with observations.
    save_fig : str or pathlib.Path
        Output figure path; parent directories are created.
    weight : array-like or None, default None
        Finite, nonnegative exposure weights with positive total; None uses row
        counts.
    feature_name : str, default 'Feature'
        Feature label used on the plot.
    n_bins : int, default 10
        Positive requested number of numeric quantile bins; duplicate edges are
        dropped.

    figure : FigureConfig or None, default None
        Optional export dimensions, DPI, and format. None preserves native
        dimensions and the requested path format.

    Returns
    -------
    pandas.DataFrame
        Columns feature, observed, fitted, and volume. Means are weighted by
        exposure; volume is group total weight. Zero-weight groups are omitted.

    Raises
    ------
    PlotError
        Values, lengths, or exposure weights are invalid.
    ValueError
        n_bins is nonpositive.

    Notes
    -----
    Creates parent directories for save_fig and closes the figure after saving.

    Examples
    --------
    >>> from pathlib import Path
    >>> from tempfile import TemporaryDirectory
    >>> from ml_toolkit.plots.regression import observed_fitted
    >>> with TemporaryDirectory() as directory:
    ...     path = Path(directory) / "observed.png"
    ...     table = observed_fitted(["a", "a"], [1.0, 3.0], [2.0, 2.0], path)
    ...     print(table["observed"].tolist(), path.is_file())
    [2.0] True
    """
    y, prediction = numeric_vector(observed, "observed"), numeric_vector(
        fitted, "fitted"
    )
    values = pd.Series(np.asarray(feature), name=feature_name)
    same_length(feature=values, observed=y, fitted=prediction)
    exposure = weights(weight, len(y))
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    numeric = pd.api.types.is_numeric_dtype(values) and values.nunique(dropna=True) > 1
    if numeric:
        groups = pd.qcut(
            values, q=min(n_bins, values.nunique(dropna=True)), duplicates="drop"
        )
        groups = groups.astype("string").fillna("missing")
    else:
        groups = values.astype("string").fillna("missing")
    grouped = _aggregate(groups, y, prediction, exposure, "feature", sort=numeric)
    x = np.arange(len(grouped))
    fig, ax = plt.subplots(figsize=(10, 6))
    volume_ax = ax.twinx()
    volume_ax.bar(x, grouped.volume, alpha=0.2, color="tab:gray")
    ax.plot(x, grouped.observed, marker="o", label="Observed")
    ax.plot(x, grouped.fitted, marker="o", label="Fitted")
    ax.set_xticks(x, grouped.feature.astype(str), rotation=45, ha="right")
    ax.set(xlabel=feature_name, ylabel="Observed / fitted")
    volume_ax.set_ylabel("Exposure / count")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, save_fig, config=figure)
    return grouped


def observed_fitted_by_level(
    reference_levels: pl.Series,
    reference_observed: ArrayLike,
    reference_fitted: ArrayLike,
    current_levels: pl.Series,
    current_observed: ArrayLike,
    current_fitted: ArrayLike,
    save_fig: str | Path,
    *,
    level_order: tuple[str, ...],
    reference_weight: ArrayLike | None = None,
    current_weight: ArrayLike | None = None,
    feature_name: str = "Feature",
    reference_label: str = "Train",
    current_label: str = "Validation",
    figure: FigureConfig | None = None,
) -> pd.DataFrame:
    """Compare weighted observed/fitted means on shared feature levels.

    Parameters
    ----------
    reference_levels, current_levels : polars.Series
        Non-null level labels produced by the same reference-fitted
        LevelTransformer. Each series aligns with its sample's values.
    reference_observed, current_observed : array-like
        Nonempty finite targets for each sample.
    reference_fitted, current_fitted : array-like
        Finite predictions aligned with each sample's targets.
    save_fig : str or pathlib.Path
        Output path; parent directories are created.
    level_order : tuple[str, ...]
        Unique fitted labels in interval/category order, including special levels.
    reference_weight, current_weight : array-like or None, default None
        Finite nonnegative weights with positive total per sample. None uses
        unit weights. Means use weights and volume is the sum of weights.
    feature_name : str, default 'Feature'
        Feature label for the horizontal axis.
    reference_label, current_label : str, default 'Train', 'Validation'
        Legend labels; use current_label='Test' for test diagnostics.
    figure : FigureConfig or None, default None
        Optional shared dimensions, DPI and export format.

    Returns
    -------
    pandas.DataFrame
        Columns sample, level, observed, fitted, volume. Samples are reference
        and current, each in level_order. Levels with zero weight in both samples
        are omitted. A zero-weight sample/level has NaN means and zero volume.

    Raises
    ------
    PlotError
        Values or weights are invalid, lengths differ, levels contain null or
        undefined labels, or level_order is empty or contains duplicates.

    Notes
    -----
    Uses supplied labels without fitting or binning. Four lines compare the two
    samples; missing means break lines. Paired bars show volume on a second axis.
    Closes the figure after saving. Appropriate target interpretation, including
    positive-unlabeled targets, is the caller's responsibility.
    """
    if not level_order or len(set(level_order)) != len(level_order):
        raise PlotError("level_order must contain unique fitted labels")
    tables: list[pd.DataFrame] = []
    for sample, levels, observed, fitted, weight in (
        (
            "reference",
            reference_levels,
            reference_observed,
            reference_fitted,
            reference_weight,
        ),
        ("current", current_levels, current_observed, current_fitted, current_weight),
    ):
        y = numeric_vector(observed, f"{sample}_observed")
        prediction = numeric_vector(fitted, f"{sample}_fitted")
        same_length(levels=levels, observed=y, fitted=prediction)
        exposure = weights(weight, len(y))
        labels = levels.cast(pl.String)
        if labels.null_count() or not labels.is_in(level_order).all():
            raise PlotError(f"{sample} contains null or undefined feature levels")
        groups = pd.Series(labels.to_list(), dtype="string")
        table = _aggregate(groups, y, prediction, exposure, "level", sort=False)
        table = table.set_index("level").reindex(pd.Index(level_order, name="level"))
        table["volume"] = table["volume"].fillna(0.0)
        table["sample"] = sample
        tables.append(table.reset_index())
    grouped = pd.concat(tables, ignore_index=True)
    active = set(grouped.loc[grouped["volume"] > 0, "level"])
    shown_levels = [level for level in level_order if level in active]
    grouped = grouped.loc[grouped["level"].isin(shown_levels)].reset_index(drop=True)
    grouped = grouped[["sample", "level", "observed", "fitted", "volume"]]

    x = np.arange(len(shown_levels))
    fig, ax = plt.subplots(figsize=(12, 6))
    volume_ax = ax.twinx()
    # Keep volume behind the observed/fitted curves.
    ax.set_zorder(volume_ax.get_zorder() + 1)
    ax.patch.set_visible(False)
    for sample, label, color, offset in (
        ("reference", reference_label, "tab:blue", -0.2),
        ("current", current_label, "tab:orange", 0.2),
    ):
        table = grouped.loc[grouped["sample"] == sample]
        volume_ax.bar(x + offset, table["volume"], width=0.4, alpha=0.18, color=color)
        ax.plot(
            x, table["observed"], marker="o", color=color, label=f"Observed — {label}"
        )
        ax.plot(
            x,
            table["fitted"],
            marker="o",
            linestyle="--",
            color=color,
            label=f"Fitted — {label}",
        )
    ax.set_xticks(x, shown_levels, rotation=45, ha="right")
    ax.set(
        xlabel=feature_name,
        ylabel="Weighted observed / fitted",
        title=f"Observed vs fitted — {feature_name}",
    )
    volume_ax.set_ylabel("Exposure / count")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, save_fig, config=figure)
    return grouped


def time_observed_fitted(
    reference_timestamp: Sequence[date | datetime],
    reference_observed: ArrayLike,
    reference_fitted: ArrayLike,
    current_timestamp: Sequence[date | datetime],
    current_observed: ArrayLike,
    current_fitted: ArrayLike,
    save_fig: str | Path,
    *,
    period: TimePeriod = "month",
    reference_weight: ArrayLike | None = None,
    current_weight: ArrayLike | None = None,
    reference_group: pl.Series | None = None,
    current_group: pl.Series | None = None,
    figure: FigureConfig | None = None,
) -> pd.DataFrame:
    """Plot time-based weighted means for two samples and optional groups.

    Parameters
    ----------
    reference_timestamp : Sequence[date | datetime]
        Dates or datetimes aligned with reference observations.
    reference_observed : array-like
        Nonempty finite reference targets.
    reference_fitted : array-like
        Finite reference predictions aligned with targets.
    current_timestamp : Sequence[date | datetime]
        Dates or datetimes aligned with current observations. Their minimum
        must be strictly greater than the maximum reference timestamp. A split
        inside one grouping period is allowed.
    current_observed : array-like
        Nonempty finite current targets.
    current_fitted : array-like
        Finite current predictions aligned with targets.
    save_fig : str or pathlib.Path
        Output figure path; parent directories are created.
    period : TimePeriod, default "month"
        Calendar grouping in matching input timezones. Minute/hour require
        datetimes; weeks start on Monday. Points are at period starts.
    reference_weight : array-like or None, default None
        Finite, nonnegative reference weights with positive total; None uses
        equal weights.
    current_weight : array-like or None, default None
        Finite, nonnegative current weights with positive total; None uses
        equal weights.
    reference_group : polars.Series or None, default None
        Optional reference group labels; None groups all rows as all.
    current_group : polars.Series or None, default None
        Optional current group labels; None groups all rows as all.

    figure : FigureConfig or None, default None
        Optional export dimensions, DPI, and format. None preserves native
        dimensions and the requested path format.

    Returns
    -------
    pandas.DataFrame
        Columns period_start, group, split, observed, fitted, and volume.
        period_start retains the input timezone; means use
        weights and zero-weight groups are omitted.

    Raises
    ------
    ValueError
        Timezones differ, or time ranges overlap or are not ordered reference
        before current.
    PlotError
        Vectors are misaligned, timestamps are missing, or numeric values and
        weights are invalid.

    Notes
    -----
    Creates parent directories for save_fig and closes the figure after saving.
    Both samples preserve their matching input timezone during calendar
    grouping. Missing periods are not filled or interpolated. Supplied group labels select separate observed/fitted curves.
    """
    reference_index = pd.DatetimeIndex(list(reference_timestamp))
    current_index = pd.DatetimeIndex(list(current_timestamp))
    if reference_index.tz != current_index.tz:
        raise ValueError(
            "Reference and current timestamps must have matching time zones"
        )
    zone = reference_index.tz
    fixed_offset = zone is not None and zone.utcoffset(None) is not None
    times: list[pl.Series] = []
    for timestamps, index in (
        (reference_timestamp, reference_index),
        (current_timestamp, current_index),
    ):
        if all(not isinstance(value, datetime) for value in timestamps):
            times.append(pl.Series("time", timestamps))
        else:
            # Fixed offsets have no DST: group their local clock values, then
            # restore the original offset. Polars only supports named zones.
            local = index.tz_localize(None) if fixed_offset else index
            times.append(pl.from_pandas(pd.Series(local, name="time")))
    reference_start, current_start = paired_period_starts(times[0], times[1], period)
    splits: list[pd.DataFrame] = []
    for name, timestamps, observed, fitted, weight, group in (
        (
            "reference",
            reference_start,
            reference_observed,
            reference_fitted,
            reference_weight,
            reference_group,
        ),
        (
            "current",
            current_start,
            current_observed,
            current_fitted,
            current_weight,
            current_group,
        ),
    ):
        y, prediction = numeric_vector(observed, f"{name}_observed"), numeric_vector(
            fitted, f"{name}_fitted"
        )
        time = pd.DatetimeIndex(timestamps.to_list())
        if fixed_offset:
            time = time.tz_localize(zone)
        same_length(timestamp=time, observed=y, fitted=prediction)
        if np.asarray(time.isna()).any():
            raise PlotError("timestamps must not contain missing values")
        exposure = weights(weight, len(y))
        labels = (
            np.full(len(y), "all", dtype=object)
            if group is None
            else np.asarray(group, dtype=object)
        )
        same_length(group=labels, observed=y)
        frame = pd.DataFrame(
            {
                "period_start": time,
                "group": pd.Series(labels).fillna("missing").astype(str),
                "observed_weighted": y * exposure,
                "fitted_weighted": prediction * exposure,
                "volume": exposure,
            }
        )
        aggregated = frame.groupby(["period_start", "group"], as_index=False).agg(
            observed_weighted=("observed_weighted", "sum"),
            fitted_weighted=("fitted_weighted", "sum"),
            volume=("volume", "sum"),
        )
        aggregated = aggregated.loc[aggregated.volume > 0].copy()
        aggregated["observed"] = aggregated.observed_weighted / aggregated.volume
        aggregated["fitted"] = aggregated.fitted_weighted / aggregated.volume
        aggregated["split"] = name
        splits.append(
            aggregated[
                ["period_start", "group", "split", "observed", "fitted", "volume"]
            ]
        )
    grouped = pd.concat(splits, ignore_index=True)
    groups = sorted(str(value) for value in grouped.group.unique())
    fig, axes = plt.subplots(len(groups), 1, figsize=(14, 5 * len(groups)), sharex=True)
    boundary_index = pd.DatetimeIndex(current_start.to_list())
    boundary = (
        boundary_index.tz_localize(zone) if fixed_offset else boundary_index
    ).min()
    for ax, label in zip(np.atleast_1d(axes), groups):
        ax.axvline(boundary, color="tab:orange", linestyle=":", label="Current begins")
        for split, linestyle in (("reference", "-"), ("current", "--")):
            subset = grouped.loc[
                (grouped.group == label) & (grouped.split == split)
            ].sort_values("period_start")
            if not subset.empty:
                ax.plot(
                    subset.period_start,
                    subset.observed,
                    color="tab:blue",
                    linestyle=linestyle,
                    marker="o",
                    label=f"Observed — {split}",
                )
                ax.plot(
                    subset.period_start,
                    subset.fitted,
                    color="tab:red",
                    linestyle=linestyle,
                    marker="o",
                    label=f"Fitted — {split}",
                )
        ax.set(title=label, ylabel="Weighted observed / fitted")
        ax.grid(alpha=0.2)
        ax.legend(loc="best")
    np.atleast_1d(axes)[-1].set_xlabel(f"Period start ({period})")
    fig.suptitle(f"Observed vs fitted ({period})")
    fig.autofmt_xdate()
    fig.tight_layout()
    save_figure(fig, save_fig, config=figure)
    return grouped
