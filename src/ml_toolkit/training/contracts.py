"""Small interface supplied by a project; the toolkit owns no data container."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, Self

import numpy as np
import pandas as pd
import polars as pl


class TrainingData(Protocol):
    """Operations used by training and the bundled model implementations.

    Projects may use their existing Pool or another object with these methods.
    ``business_type`` is optional and is read only when available.
    """

    X: pl.DataFrame
    y: pl.Series
    time: pl.Series
    weight: pl.Series
    cat_features: tuple[str, ...]

    def __len__(self) -> int: ...

    def take(self, indices: Sequence[int] | np.ndarray) -> Self: ...

    def select_features(self, feature_names: Sequence[str]) -> Self: ...

    def with_target(self, target: pl.Series | np.ndarray) -> Self: ...

    def to_catboost(self, baseline: float | np.ndarray | None = None) -> Any: ...

    def to_model_input(self, n_rows: int | None = None) -> pd.DataFrame: ...


def validate_training_data(data: TrainingData) -> None:
    """Check only invariants needed by chronological model training."""
    size = len(data)
    if size == 0 or len(data.X) != size or any(
        len(series) != size for series in (data.y, data.time, data.weight)
    ):
        raise ValueError("Features, target, time, and weight need equal nonzero lengths")
    if data.time.null_count() or not data.time.is_sorted():
        raise ValueError("Training data must have non-null, ascending timestamps")
    target = np.asarray(data.y.to_numpy(), dtype=float)
    weight = np.asarray(data.weight.to_numpy(), dtype=float)
    if not np.isfinite(target).all():
        raise ValueError("Target values must be finite")
    if not np.isfinite(weight).all() or (weight < 0).any() or weight.sum() <= 0:
        raise ValueError("Weights must be finite, nonnegative, and have positive total")
