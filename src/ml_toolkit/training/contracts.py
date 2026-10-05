"""Shared interface for toolkit Pool and compatible project data containers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, Self

import numpy as np
import pandas as pd
import polars as pl


class TrainingData(Protocol):
    """Operations used by training and the bundled model implementations.

    Projects may use ml_toolkit.data.Pool or their own compatible container.
    ``business_type`` is optional and is read only when available.
    """

    @property
    def X(self) -> pl.DataFrame:
        """Return aligned feature columns."""
        ...

    @property
    def y(self) -> pl.Series:
        """Return aligned target values."""
        ...

    @property
    def time(self) -> pl.Series:
        """Return aligned observation timestamps."""
        ...

    @property
    def weight(self) -> pl.Series:
        """Return aligned row weights."""
        ...

    @property
    def cat_features(self) -> tuple[str, ...]:
        """Return categorical feature names."""
        ...

    def __len__(self) -> int: ...

    def take(self, indices: Sequence[int] | np.ndarray) -> Self: ...

    def select_features(self, feature_names: Sequence[str]) -> Self: ...

    def with_target(self, target: pl.Series | np.ndarray) -> Self: ...

    def to_catboost(self, baseline: float | np.ndarray | None = None) -> Any: ...

    def to_model_input(self, n_rows: int | None = None) -> pd.DataFrame:
        """Return model input; toolkit Pool limits to first rows for timestamp alignment."""
        ...


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
