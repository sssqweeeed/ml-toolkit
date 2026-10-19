"""Data validation at training and model boundaries."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from .training.contracts import Pool


def validate_object_ids(object_id: pl.Series, n_rows: int) -> None:
    if len(object_id) != n_rows:
        raise ValueError("object_id must have equal lengths with data rows")
    if object_id.dtype != pl.String and not object_id.dtype.is_integer():
        raise ValueError("object_id must have string or integer dtype")
    if object_id.null_count():
        raise ValueError("object_id must not contain null values")


def validate_training_data(data: Pool) -> None:
    size = len(data)
    series = [data.y, data.time, data.weight]
    if data.business_type is not None:
        series.append(data.business_type)
    if (
        size == 0
        or len(data.X) != size
        or any(len(column) != size for column in series)
    ):
        raise ValueError(
            "Features, target, time, and weight need equal nonzero lengths"
        )
    if data.time.dtype != pl.Date and data.time.dtype.base_type() != pl.Datetime:
        raise ValueError("Training time must have Date or Datetime dtype")
    if data.time.null_count() or not data.time.is_sorted():
        raise ValueError("Training data must have non-null, ascending timestamps")
    target = data.y.to_numpy().astype(float)
    weight = data.weight.to_numpy().astype(float)
    if not np.isfinite(target).all():
        raise ValueError("Target values must be finite")
    if not np.isfinite(weight).all() or (weight < 0).any() or weight.sum() <= 0:
        raise ValueError("Weights must be finite, nonnegative, and have positive total")
    if data.object_id is not None:
        validate_object_ids(data.object_id, size)


def validate_training_categories(data: Pool) -> None:
    for name in data.cat_features:
        if name not in data.X.columns:
            raise ValueError(f"Categorical feature is missing: {name}")
        if (data.X[name].cast(pl.String) == "unknown").any():
            raise ValueError(
                f"Categorical feature {name!r} contains reserved value 'unknown'"
            )
