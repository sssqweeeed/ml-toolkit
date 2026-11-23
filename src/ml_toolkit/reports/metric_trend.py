"""Task-metric windows on recorded calendar periods."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from .._time import PERIOD_INTERVALS, period_starts
from ..data import Pool, validate_prediction, validate_split_labels
from ..feature_levels import FeatureLevelConfig
from ..figures import FigureConfig, PlotError, feature_filename, saved_figure
from ..metrics import Metric
from ..types import FloatArray, IntArray, TimePeriod
from .base import PredictionEvaluator, expand_skipped_rows, long_metric_rows

__all__ = ["MetricTrend", "metric_trend"]


@dataclass(frozen=True, kw_only=True)
class MetricTrend(PredictionEvaluator):
    """Calculate task metrics in separate calendar windows for each split.

    Parameters
    ----------
    metric : Metric
        Task callback accepting aligned target, prediction and weight arrays and
        returning named scalar scores through to_dict(). Use the same callback
        as train_model when the report should show its training metrics.
    period : TimePeriod, default 'month'
        Calendar bucket from minute through year, in the pool's timezone.
        Minute and hour require Datetime values. Weeks start on Monday.
    window : int, default 1
        Positive number of calendar buckets ending at each observed bucket.
        For example, window=3 with period='month' includes the current month and
        the preceding two months. Initial windows use the available rows.
    min_rows : int, default 1
        Minimum number of positive-weight rows required to call the metric.
    name : str, default 'metric_trend'
        Unique evaluator name and artifact namespace.

    Notes
    -----
    Calculates every named score independently on train and at most one held-out
    split. Rolling windows never cross split boundaries. Each observed period
    gets one endpoint; missing calendar periods are not synthesized. Zero-weight
    and undersupported windows retain support counts and null scores. Undefined
    scores are null/undefined; metric exceptions propagate with window context.

    calculate() returns a long Polars table. write_artifacts() calculates once,
    writes metrics.parquet, and draws each metric with a window-weight panel.
    The inherited evaluate() returns no additional scalar scores. Settings may
    be reused across fits; data and predictions are never changed or retained.
    """

    metric: Metric
    period: TimePeriod = "month"
    window: int = 1
    min_rows: int = 1
    name: str = "metric_trend"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.period not in PERIOD_INTERVALS:
            raise ValueError("Unsupported time period")
        if type(self.window) is not int or self.window < 1:
            raise ValueError("window must be a positive integer")
        if type(self.min_rows) is not int or self.min_rows < 1:
            raise ValueError("min_rows must be a positive integer")

    def parameters(self) -> dict[str, object]:
        metric_type = type(self.metric)
        identifier = (
            f"{getattr(self.metric, '__module__', metric_type.__module__)}."
            f"{getattr(self.metric, '__qualname__', metric_type.__qualname__)}"
        )
        return {
            "metric": identifier,
            "period": self.period,
            "window": self.window,
            "min_rows": self.min_rows,
        }

    def calculate(
        self, *, data: Pool, prediction: FloatArray, split: pl.Series
    ) -> pl.DataFrame:
        """Return weighted scores and support for each observed split/window.

        Parameters
        ----------
        data : Pool
            Aligned rows with finite targets, recorded Date/Datetime timestamps,
            and finite nonnegative weights. Zero-total windows are retained.
        prediction : FloatArray
            Finite float predictions with shape (n_rows,), in task target units.
        split : polars.Series
            Aligned non-null String labels containing train and at most one of
            val/test. Held-out-only and train-only inputs are also supported.

        Returns
        -------
        polars.DataFrame
            Long table with period_start and window_start retaining the calendar
            timezone; split; n_rows; n_positive_weight_rows; sum_weight; metric;
            nullable value; and status (ok, undefined, zero_weight, or
            insufficient_rows). Skipped windows expand to known metric names;
            metric stays null if no window could establish those names.

        Raises
        ------
        ValueError
            Inputs are misaligned, nonfinite, have negative weights, or metric
            names are empty or change between windows. Callback errors propagate
            with evaluator, split, and endpoint context.
        """
        validate_prediction(data, prediction)
        validate_split_labels(data, split)
        target = data.y.to_numpy().astype(float)
        weight = data.weight.to_numpy().astype(float)
        if not np.isfinite(target).all():
            raise ValueError("Target values must be finite")
        if not np.isfinite(weight).all() or (weight < 0).any():
            raise ValueError("Weights must be finite and nonnegative")
        starts = period_starts(data.time, self.period)
        instants: IntArray = starts.cast(pl.Int64).to_numpy()
        offset = f"-{self.window - 1}{PERIOD_INTERVALS[self.period][1:]}"
        records: list[dict[str, object]] = []
        names: set[str] | None = None
        for part in ("train", "val", "test"):
            selected = (split == part).to_numpy()
            if not selected.any():
                continue
            endpoints = starts.filter(pl.Series(selected)).unique().sort()
            lower = endpoints.dt.offset_by(offset)
            first_instants: IntArray = lower.cast(pl.Int64).to_numpy()
            last_instants: IntArray = endpoints.cast(pl.Int64).to_numpy()
            positions = np.flatnonzero(selected)
            order = np.argsort(instants[positions], kind="stable")
            sorted_index = positions[order]
            sorted_instants: IntArray = instants[sorted_index]
            for position in range(len(endpoints)):
                endpoint = endpoints[position]
                try:
                    left = int(
                        sorted_instants.searchsorted(
                            int(first_instants[position]), side="left"
                        )
                    )
                    right = int(
                        sorted_instants.searchsorted(
                            int(last_instants[position]), side="right"
                        )
                    )
                    index = np.sort(sorted_index[left:right])
                    positive = int(np.count_nonzero(weight[index] > 0))
                    total = float(weight[index].sum())
                    status = (
                        "zero_weight"
                        if positive == 0
                        else "insufficient_rows"
                        if positive < self.min_rows
                        else "ok"
                    )
                    base: dict[str, object] = {
                        "period_start": endpoint,
                        "window_start": lower[position],
                        "split": part,
                        "n_rows": len(index),
                        "n_positive_weight_rows": positive,
                        "sum_weight": total,
                    }
                    if status != "ok":
                        records.append(
                            {**base, "metric": None, "value": None, "status": status}
                        )
                        continue
                    scores = self.metric(
                        target[index], prediction[index], weight[index]
                    ).to_dict()
                    rows = long_metric_rows(base, scores)
                    if names is None:
                        names = set(scores)
                    elif names != set(scores):
                        raise ValueError("Metric names must match across windows")
                    records.extend(rows)
                except Exception as exc:
                    exc.add_note(
                        f"Metric trend evaluator={self.name}, split={part}, period_start={endpoint}"
                    )
                    raise
        return pl.DataFrame(
            expand_skipped_rows(records, names or set()),
            schema={
                "period_start": starts.dtype,
                "window_start": starts.dtype,
                "split": pl.String,
                "n_rows": pl.Int64,
                "n_positive_weight_rows": pl.Int64,
                "sum_weight": pl.Float64,
                "metric": pl.String,
                "value": pl.Float64,
                "status": pl.String,
            },
        )

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
        table = self.calculate(data=data, prediction=prediction, split=split)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / "metrics.parquet"
        table.write_parquet(path)
        paths = [path]
        for index, key in enumerate(
            table["metric"].drop_nulls().unique(maintain_order=True).to_list()
        ):
            path = artifact_dir / feature_filename(index, key, figure.format)
            metric_trend(
                table, metric=key, period=self.period, save_fig=path, figure=figure
            )
            paths.append(path)
        return paths


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
    with saved_figure(fig, save_fig, config=figure):
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
