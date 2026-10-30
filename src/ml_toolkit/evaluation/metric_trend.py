"""Task-metric windows on recorded calendar periods."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from .base import PredictionEvaluator
from .._time import PERIOD_INTERVALS, period_starts
from ..config import FigureConfig
from ..data import Pool
from ..data_analysis._common import validate_split
from ..feature_levels import FeatureLevelConfig
from ..metrics import Metric
from ..metrics._evaluation import validate_prediction
from ..plots.utils import feature_filename
from ..types import FloatArray, IntArray, TimePeriod


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
        validate_split(data, split)
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
                        else "insufficient_rows" if positive < self.min_rows else "ok"
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
                    if not scores or any(
                        type(key) is not str or not key.strip() for key in scores
                    ):
                        raise ValueError("Metric must return nonempty named scores")
                    if names is None:
                        names = set(scores)
                    elif names != set(scores):
                        raise ValueError("Metric names must match across windows")
                    for key, score in scores.items():
                        value = float(score)
                        finite = bool(np.isfinite(value))
                        records.append(
                            {
                                **base,
                                "metric": key,
                                "value": value if finite else None,
                                "status": "ok" if finite else "undefined",
                            }
                        )
                except Exception as exc:
                    exc.add_note(
                        f"Metric trend evaluator={self.name}, split={part}, period_start={endpoint}"
                    )
                    raise
        expanded: list[dict[str, object]] = []
        for record in records:
            if record["metric"] is None and names:
                expanded.extend({**record, "metric": name} for name in sorted(names))
            else:
                expanded.append(record)
        return pl.DataFrame(
            expanded,
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
        from ..plots.metric_trend import metric_trend

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
