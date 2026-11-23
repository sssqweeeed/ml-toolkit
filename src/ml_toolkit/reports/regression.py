"""Observed-versus-fitted reports by feature level and calendar period."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from numpy.typing import ArrayLike

from .._time import PERIOD_INTERVALS, paired_period_starts
from ..data import Pool
from ..feature_levels import FeatureLevelConfig, LevelTransformer
from ..feature_levels.config import feature_definitions, validate_features
from ..figures import (
    FigureConfig,
    PlotError,
    feature_filename,
    numeric_vector,
    same_length,
    saved_figure,
    weights,
)
from ..types import FloatArray, TimePeriod
from .base import PredictionEvaluator, comparison_pair
from .grouped import row_groups, validate_grouping

__all__ = [
    "ObservedFitted",
    "TimeObservedFitted",
    "TimeSummary",
    "level_observed_fitted",
    "observed_fitted_by_level",
    "time_observed_fitted",
    "time_summary",
]


@dataclass(frozen=True, kw_only=True)
class ObservedFitted(PredictionEvaluator):
    """Compare weighted observed/fitted means on train-fitted feature levels.

    Parameters
    ----------
    name : str, default 'observed_fitted'
        Unique report name and artifact namespace.
    features : tuple of str or None, default None
        Names to display from shared feature definitions. None selects all;
        an empty tuple selects none. Unknown names raise ValueError.

    Notes
    -----
    Fits LevelTransformer on train using the supplied feature rules and n_bins.
    Each figure compares train with validation or test using the same levels.
    Lines show weighted means; volume is summed weight. Zero-weight or absent
    levels have no mean. Saves one indexed feature filename in the report root.
    """

    name: str = "observed_fitted"
    features: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        validate_features(self.features)

    def parameters(self) -> dict[str, object]:
        return {"features": self.features}

    def write_artifacts(
        self,
        *,
        data: Pool,
        prediction: FloatArray,
        split: pl.Series,
        artifact_dir: Path,
        figure: FigureConfig,
        features: tuple[FeatureLevelConfig, ...] | None,
        n_bins: int,
    ) -> list[Path]:
        paths: list[Path] = []
        pair = comparison_pair(data, prediction, split)
        if pair is None:
            return []
        part, train, train_pred, current, current_pred = pair
        definitions = feature_definitions(train, features)
        if not definitions:
            return []
        if self.features is not None:
            if not set(self.features) <= {item.name for item in definitions}:
                raise ValueError(
                    "ObservedFitted features must exist in shared feature definitions"
                )
            by_name = {item.name: item for item in definitions}
            definitions = tuple(by_name[name] for name in self.features)
        if not definitions:
            return []
        transformer = LevelTransformer(definitions, n_bins=n_bins).fit(
            train.to_analysis_frame()
        )
        reference_levels = transformer.transform(train.to_analysis_frame())
        current_levels = transformer.transform(current.to_analysis_frame())
        assert transformer.feature_levels is not None
        for index, level in enumerate(transformer.feature_levels):
            path = artifact_dir / feature_filename(
                index, level.config.name, figure.format
            )
            observed_fitted_by_level(
                reference_levels[level.config.name],
                train.y.to_numpy(),
                train_pred,
                current_levels[level.config.name],
                current.y.to_numpy(),
                current_pred,
                save_fig=path,
                level_order=level.levels,
                reference_weight=train.weight.to_numpy(),
                current_weight=current.weight.to_numpy(),
                feature_name=level.config.name,
                current_label="Test" if part == "test" else "Validation",
                figure=figure,
            )
            paths.append(path)
        return paths


@dataclass(frozen=True, kw_only=True)
class TimeObservedFitted(PredictionEvaluator):
    """Compare weighted time summaries, optionally grouped by exact values.

    Parameters
    ----------
    name : str, default 'time_observed_fitted'
        Unique report name and artifact namespace.
    period : TimePeriod, default 'month'
        Calendar grouping from minute through year, in the data timezone.
    by : tuple of str, default ()
        Model or auxiliary columns defining exact-value groups. Multiple columns
        select combinations. Empty gives an overall plot. Precompute bands for
        continuous values; no numerical levels are fitted here.
    max_groups : int, default 50
        Maximum distinct groups across both samples, including missing groups.

    Notes
    -----
    Compares train with validation or test, with a panel for each requested group.
    Observed/fitted means and volumes use row weights.
    Timestamps must be ordered across the pair; a shared endpoint is allowed.
    Saves ``time.<format>``. Shared feature rules and n_bins are unused.
    """

    name: str = "time_observed_fitted"
    period: TimePeriod = "month"
    by: tuple[str, ...] = ()
    max_groups: int = 50

    def __post_init__(self) -> None:
        super().__post_init__()
        validate_grouping(self.by, self.max_groups, required=False)
        if self.period not in PERIOD_INTERVALS:
            raise ValueError("Unsupported time period")

    def parameters(self) -> dict[str, object]:
        return {"period": self.period, "by": self.by, "max_groups": self.max_groups}

    def write_artifacts(
        self,
        *,
        data: Pool,
        prediction: FloatArray,
        split: pl.Series,
        artifact_dir: Path,
        figure: FigureConfig,
        features: tuple[FeatureLevelConfig, ...] | None,
        n_bins: int,
    ) -> list[Path]:
        pair = comparison_pair(data, prediction, split)
        if pair is None:
            return []
        _, train, train_pred, current, current_pred = pair
        reference_group = current_group = None
        if self.by:
            labels = [""] * len(data)
            for group in row_groups(data, self.by, self.max_groups):
                for index in group.indices:
                    labels[int(index)] = group.label
            groups = pl.Series(labels)
            reference_group = groups.filter(split == "train")
            current_group = groups.filter(split != "train")
        path = artifact_dir / f"time.{figure.format}"
        time_observed_fitted(
            train.time.to_list(),
            train.y.to_numpy(),
            train_pred,
            current.time.to_list(),
            current.y.to_numpy(),
            current_pred,
            save_fig=path,
            reference_weight=train.weight.to_numpy(),
            current_weight=current.weight.to_numpy(),
            reference_group=reference_group,
            current_group=current_group,
            period=self.period,
            figure=figure,
        )
        return [path]


def _aggregate(frame: pd.DataFrame, groups: list[str], *, sort: bool) -> pd.DataFrame:
    """Aggregate weighted outcomes for feature levels or calendar groups."""
    grouped = frame.groupby(groups, sort=sort, as_index=False, observed=True).agg(
        observed_weighted=("observed_weighted", "sum"),
        fitted_weighted=("fitted_weighted", "sum"),
        volume=("volume", "sum"),
    )
    grouped = grouped.loc[grouped.volume > 0].copy()
    grouped["observed"] = grouped.observed_weighted / grouped.volume
    grouped["fitted"] = grouped.fitted_weighted / grouped.volume
    return grouped[[*groups, "observed", "fitted", "volume"]]


def level_observed_fitted(
    reference_levels: pl.Series,
    reference_observed: ArrayLike,
    reference_fitted: ArrayLike,
    current_levels: pl.Series,
    current_observed: ArrayLike,
    current_fitted: ArrayLike,
    *,
    level_order: tuple[str, ...],
    reference_weight: ArrayLike | None = None,
    current_weight: ArrayLike | None = None,
) -> pd.DataFrame:
    """Weighted observed/fitted means per sample and level, without drawing.

    Parameters
    ----------
    reference_levels, current_levels : polars.Series
        Non-null level labels from the same fitted LevelTransformer.
    reference_observed, current_observed : array-like
        Nonempty finite targets for each sample.
    reference_fitted, current_fitted : array-like
        Finite predictions aligned with each sample's targets.
    level_order : tuple of str
        Unique fitted labels in interval/category order.
    reference_weight, current_weight : array-like or None, default None
        Finite nonnegative weights with positive total; None uses unit weights.

    Returns
    -------
    pandas.DataFrame
        Columns sample, level, observed, fitted and volume, as returned by
        :func:`observed_fitted_by_level`.

    Raises
    ------
    PlotError
        Values or weights are invalid, lengths differ, levels are null or
        undefined, or ``level_order`` is empty or has duplicates.
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
        table = _aggregate(
            pd.DataFrame(
                {
                    "level": groups,
                    "observed_weighted": y * exposure,
                    "fitted_weighted": prediction * exposure,
                    "volume": exposure,
                }
            ),
            ["level"],
            sort=False,
        )
        table = table.set_index("level").reindex(pd.Index(level_order, name="level"))
        table["volume"] = table["volume"].fillna(0.0)
        table["sample"] = sample
        tables.append(table.reset_index())
    grouped = pd.concat(tables, ignore_index=True)
    active = set(grouped.loc[grouped["volume"] > 0, "level"])
    shown_levels = [level for level in level_order if level in active]
    grouped = grouped.loc[grouped["level"].isin(shown_levels)].reset_index(drop=True)
    grouped = grouped[["sample", "level", "observed", "fitted", "volume"]]
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
    grouped = level_observed_fitted(
        reference_levels,
        reference_observed,
        reference_fitted,
        current_levels,
        current_observed,
        current_fitted,
        level_order=level_order,
        reference_weight=reference_weight,
        current_weight=current_weight,
    )
    active = set(grouped["level"])
    shown_levels = [level for level in level_order if level in active]
    x = np.arange(len(shown_levels))
    fig, ax = plt.subplots(figsize=(12, 6))
    with saved_figure(fig, save_fig, config=figure):
        volume_ax = ax.twinx()
        # Keep volume behind the observed/fitted curves.
        ax.set_zorder(volume_ax.get_zorder() + 1)
        ax.patch.set_visible(False)
        for sample, label, color, offset in (
            ("reference", reference_label, "tab:blue", -0.2),
            ("current", current_label, "tab:orange", 0.2),
        ):
            table = grouped.loc[grouped["sample"] == sample]
            volume_ax.bar(
                x + offset, table["volume"], width=0.4, alpha=0.18, color=color
            )
            ax.plot(
                x,
                table["observed"],
                marker="o",
                color=color,
                label=f"Observed — {label}",
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
    return grouped


@dataclass(frozen=True)
class TimeSummary:
    """Weighted period means and the timestamp where the current sample begins.

    Attributes
    ----------
    table : pandas.DataFrame
        Columns period_start, group, split, observed, fitted and volume.
    boundary : pandas.Timestamp
        Earliest period start of the current sample, in the input timezone.
    """

    table: pd.DataFrame
    boundary: pd.Timestamp


def time_summary(
    reference_timestamp: Sequence[date | datetime],
    reference_observed: ArrayLike,
    reference_fitted: ArrayLike,
    current_timestamp: Sequence[date | datetime],
    current_observed: ArrayLike,
    current_fitted: ArrayLike,
    *,
    period: TimePeriod = "month",
    reference_weight: ArrayLike | None = None,
    current_weight: ArrayLike | None = None,
    reference_group: pl.Series | None = None,
    current_group: pl.Series | None = None,
) -> TimeSummary:
    """Weighted observed/fitted means per period and group, without drawing.

    Parameters
    ----------
    reference_timestamp, current_timestamp : Sequence of date or datetime
        Timestamps aligned with each sample; the earliest current timestamp must
        not precede the latest reference timestamp.
    reference_observed, current_observed : array-like
        Nonempty finite targets.
    reference_fitted, current_fitted : array-like
        Finite predictions aligned with targets.
    period : TimePeriod, default 'month'
        Calendar grouping in the matching input timezone.
    reference_weight, current_weight : array-like or None, default None
        Finite nonnegative weights with positive total; None uses unit weights.
    reference_group, current_group : polars.Series or None, default None
        Optional group labels; None groups all rows as ``all``.

    Returns
    -------
    TimeSummary
        The table described by :func:`time_observed_fitted` and its boundary.

    Raises
    ------
    ValueError
        Timezones differ or the reference extends past the current sample.
    PlotError
        Vectors are misaligned, timestamps are missing, or values are invalid.
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
        y, prediction = (
            numeric_vector(observed, f"{name}_observed"),
            numeric_vector(fitted, f"{name}_fitted"),
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
        aggregated = _aggregate(frame, ["period_start", "group"], sort=True)
        aggregated["split"] = name
        splits.append(
            aggregated[
                ["period_start", "group", "split", "observed", "fitted", "volume"]
            ]
        )
    grouped = pd.concat(splits, ignore_index=True)
    boundary_index = pd.DatetimeIndex(current_start.to_list())
    boundary = (
        boundary_index.tz_localize(zone) if fixed_offset else boundary_index
    ).min()
    return TimeSummary(grouped, boundary)


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
        must be greater than or equal to the maximum reference timestamp. A split
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
        Timezones differ, or the latest reference timestamp exceeds the earliest
        current timestamp.
    PlotError
        Vectors are misaligned, timestamps are missing, or numeric values and
        weights are invalid.

    Notes
    -----
    Creates parent directories for save_fig and closes the figure after saving.
    Both samples preserve their matching input timezone during calendar
    grouping. Missing periods are not filled or interpolated. Supplied group labels select separate observed/fitted curves.
    """
    summary = time_summary(
        reference_timestamp,
        reference_observed,
        reference_fitted,
        current_timestamp,
        current_observed,
        current_fitted,
        period=period,
        reference_weight=reference_weight,
        current_weight=current_weight,
        reference_group=reference_group,
        current_group=current_group,
    )
    grouped, boundary = summary.table, summary.boundary
    groups = sorted(str(value) for value in grouped.group.unique())
    fig, axes = plt.subplots(len(groups), 1, figsize=(14, 5 * len(groups)), sharex=True)
    with saved_figure(fig, save_fig, config=figure):
        for ax, label in zip(np.atleast_1d(axes), groups):
            ax.axvline(
                boundary, color="tab:orange", linestyle=":", label="Current begins"
            )
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
    return grouped
