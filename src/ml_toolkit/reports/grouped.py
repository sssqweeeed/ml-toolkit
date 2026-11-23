"""Task metrics calculated for exact-value groups and data splits."""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from ..data import Pool
from ..feature_levels import FeatureLevelConfig
from ..feature_levels.config import validate_features
from ..figures import FigureConfig
from ..metrics import Metric
from ..types import FloatArray, IntArray
from .base import (
    PredictionEvaluator,
    expand_skipped_rows,
    long_metric_rows,
    validate_inputs,
)

__all__ = ["GroupedMetrics", "RowGroup", "row_groups", "validate_grouping"]


@dataclass(frozen=True)
class RowGroup:
    key: pl.DataFrame
    indices: IntArray
    identifier: str
    label: str


def validate_grouping(by: tuple[str, ...], max_groups: int, *, required: bool) -> None:
    validate_features(by)
    if required and not by:
        raise ValueError("by must contain at least one grouping column")
    if type(max_groups) is not int or max_groups < 1:
        raise ValueError("max_groups must be a positive integer")


def row_groups(data: Pool, by: tuple[str, ...], max_groups: int) -> list[RowGroup]:
    """Partition positions by typed values; null/NaN form one missing group.

    IDs are ordinal within a report and never derived from user labels. Labels
    include that ID so different typed keys cannot collapse after formatting.
    """
    frame = data.to_analysis_frame(by)
    if any(dtype.is_nested() or dtype == pl.Object for dtype in frame.schema.values()):
        raise ValueError("Grouping columns must contain scalar values")
    frame = frame.with_columns(
        pl.col(name).fill_nan(None)
        for name, dtype in frame.schema.items()
        if dtype.is_float()
    )
    # Check cardinality before materializing a potentially huge list of groups.
    if frame.n_unique() > max_groups:
        raise ValueError(
            f"Grouping exceeds max_groups={max_groups}; use coarser groups or precomputed bands"
        )
    position = "__ml_toolkit_group_row__"
    while position in frame.columns:
        position += "_"
    parts = frame.with_row_index(position).partition_by(list(by), maintain_order=True)
    groups: list[RowGroup] = []
    for index, part in enumerate(parts):
        key = part.select(by).head(1)
        identifier = f"group_{index:04d}"
        description = ", ".join(
            f"{name}={value!r}" for name, value in key.row(0, named=True).items()
        )
        groups.append(
            RowGroup(
                key=key,
                indices=part[position].to_numpy().astype(np.int64),
                identifier=identifier,
                label=f"{identifier}: {description}",
            )
        )
    return groups


@dataclass(frozen=True, kw_only=True)
class GroupedMetrics(PredictionEvaluator):
    """Calculate an ordinary metric separately for exact-value groups and splits.

    Parameters
    ----------
    metric : Metric
        Callable accepting aligned target, prediction and weight arrays and
        returning an object with to_dict(). Exceptions propagate unchanged.
    by : tuple of str
        Nonempty model/auxiliary grouping columns. Multiple names select
        combinations. Prepare a categorical band column for continuous values.
    name : str, default 'grouped_metrics'
        Unique report name and artifact namespace.
    min_rows : int, default 1
        Minimum number of positive-weight rows required to calculate metrics.
    max_groups : int, default 50
        Maximum distinct groups across all supplied splits, including missing.

    Notes
    -----
    calculate() returns a typed long table, including empty split/group pairs.
    Statuses are ok, empty, zero_weight, insufficient_rows, or undefined.
    Missing results have null values. Only finite results are logged by evaluate.
    write_artifacts saves metrics.parquet, preserving typed group values and IDs.
    Instances retain settings only and can be reused across folds.
    """

    metric: Metric
    by: tuple[str, ...]
    name: str = "grouped_metrics"
    min_rows: int = 1
    max_groups: int = 50

    def __post_init__(self) -> None:
        super().__post_init__()
        validate_grouping(self.by, self.max_groups, required=True)
        if type(self.min_rows) is not int or self.min_rows < 1:
            raise ValueError("min_rows must be a positive integer")
        reserved = {
            "split",
            "group_id",
            "n_rows",
            "n_positive_weight_rows",
            "sum_weight",
            "metric",
            "value",
            "status",
        }
        if reserved.intersection(self.by):
            raise ValueError("Grouping column names conflict with metric table columns")

    def parameters(self) -> dict[str, object]:
        metric_type = type(self.metric)
        identifier = (
            f"{getattr(self.metric, '__module__', metric_type.__module__)}."
            f"{getattr(self.metric, '__qualname__', metric_type.__qualname__)}"
        )
        return {
            "metric": identifier,
            "by": self.by,
            "min_rows": self.min_rows,
            "max_groups": self.max_groups,
        }

    def calculate(
        self, *, data: Pool, prediction: FloatArray, split: pl.Series
    ) -> pl.DataFrame:
        """Return per-split/group scores with typed keys, support counts and status.

        Groups are determined across the supplied rows. Each represented split
        receives every group, including those with no rows in that split.
        Skipped groups have metric=None when no metric names can be determined.
        Nonfinite scores become null/undefined; metric exceptions are not hidden.
        """
        validate_inputs(data, prediction, split)
        groups = row_groups(data, self.by, self.max_groups)
        target = data.y.to_numpy().astype(float)
        weight = data.weight.to_numpy().astype(float)
        if not np.isfinite(target).all():
            raise ValueError("Target values must be finite")
        if not np.isfinite(weight).all() or (weight < 0).any():
            raise ValueError("Weights must be finite and nonnegative")
        records: list[dict[str, object]] = []
        metric_names: set[str] = set()
        for part in ("train", "val", "test"):
            selected = (split == part).to_numpy()
            if not selected.any():
                continue
            for group in groups:
                index = group.indices[selected[group.indices]]
                positive = int(np.count_nonzero(weight[index] > 0))
                total = float(weight[index].sum())
                if not len(index):
                    status = "empty"
                elif positive == 0:
                    status = "zero_weight"
                elif positive < self.min_rows:
                    status = "insufficient_rows"
                else:
                    status = "ok"
                base: dict[str, object] = {
                    **group.key.row(0, named=True),
                    "group_id": group.identifier,
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
                metric_names.update(str(row["metric"]) for row in rows)
                records.extend(rows)
        expanded = expand_skipped_rows(records, metric_names)
        schema = dict(data.to_analysis_frame(self.by).schema)
        schema.update(
            {
                "group_id": pl.String(),
                "split": pl.String(),
                "n_rows": pl.Int64(),
                "n_positive_weight_rows": pl.Int64(),
                "sum_weight": pl.Float64(),
                "metric": pl.String(),
                "value": pl.Float64(),
                "status": pl.String(),
            }
        )
        return pl.DataFrame(expanded, schema=schema)

    def evaluate(
        self, *, data: Pool, prediction: FloatArray, split: pl.Series
    ) -> dict[str, float]:
        table = self.calculate(data=data, prediction=prediction, split=split)
        scores: dict[str, float] = {}
        for row in table.filter(pl.col("status") == "ok").iter_rows(named=True):
            name = str(row["metric"])
            # Ordinary names stay readable; hex encoding is injective and uses
            # a reserved leading underscore, so unsafe names cannot collide.
            key = (
                name
                if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)
                else "_hex_" + name.encode().hex()
            )
            scores[f"{row['split']}/{row['group_id']}/{key}"] = float(row["value"])
        return scores

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
        return [path]
