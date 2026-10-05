"""Shared interface for toolkit Pool and compatible project data containers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, Self

import numpy as np
import pandas as pd
import polars as pl


class TrainingData(Protocol):
    """Describe the data interface used by training and models.

    Attributes
    ----------
    X : polars.DataFrame
        Feature columns with one row per observation.
    y : polars.Series
        Finite numeric target values aligned with X.
    time : polars.Series
        Non-null ascending observation timestamps aligned with X.
    weight : polars.Series
        Finite, nonnegative row weights with positive total.
    cat_features : tuple[str, ...]
        Names of categorical columns in X.

    Notes
    -----
    Projects may use ml_toolkit.data.Pool or their own compatible container.
    Row operations must keep features, target, timestamps, and weights aligned.
    Optional business_type values are used for grouped monthly plots. Custom
    models need only the data operations their workflows invoke. Attribute
    access is read-only here so frozen Pool and mutable custom containers both
    satisfy the protocol.
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

    def __len__(self) -> int:
        """Return the number of aligned observations."""
        ...

    def take(self, indices: Sequence[int] | np.ndarray) -> Self:
        """Return aligned rows at positional indices.

        Parameters
        ----------
        indices : Sequence[int] or numpy.ndarray
            Row positions to retain, in the requested order.

        Returns
        -------
        Self
            Project data containing the selected rows and their metadata.
        """
        ...

    def select_features(self, feature_names: Sequence[str]) -> Self:
        """Return data restricted to the requested feature columns.

        Parameters
        ----------
        feature_names : Sequence[str]
            Names and order of retained feature columns.

        Returns
        -------
        Self
            Data with unchanged rows, targets, timestamps, and weights; categorical
            metadata must reflect the retained features.
        """
        ...

    def with_target(self, target: pl.Series | np.ndarray) -> Self:
        """Return data with a replacement target and unchanged feature rows.

        Parameters
        ----------
        target : polars.Series or numpy.ndarray
            Replacement values aligned one-to-one with the existing rows.

        Returns
        -------
        Self
            Project data containing the new target.
        """
        ...

    def to_catboost(self, baseline: float | np.ndarray | None = None) -> Any:
        """Convert data to a CatBoost Pool for the bundled model adapters.

        Parameters
        ----------
        baseline : float, numpy.ndarray, or None, default None
            Optional raw prediction baseline, broadcast or aligned per row by the
            project implementation.

        Returns
        -------
        catboost.Pool
            Pool containing features, labels, weights, categorical metadata, and
            the optional baseline.
        """
        ...

    def to_model_input(self, n_rows: int | None = None) -> pd.DataFrame:
        """Convert feature rows to a pandas serving input frame.

        Parameters
        ----------
        n_rows : int or None, default None
            Optional limit on the number of rows in the serving example. The
            toolkit Pool returns the first rows, preserving timestamp alignment.

        Returns
        -------
        pandas.DataFrame
            Model input with named feature columns; categorical values must be
            compatible with the native model.
        """
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
