"""Shared Polars data container for training and bundled model adapters."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import catboost
import numpy as np
import pandas as pd
import polars as pl


@dataclass(frozen=True)
class Pool:
    """Hold aligned features, targets, timestamps, weights, and row metadata.

    Projects prepare the data and may optionally supply ``business_type`` for
    grouped plots. Feature lists may describe only a subset of X's columns.
    Fields cannot be reassigned; the contained frames and series are not deeply
    immutable. Pool methods return new objects without modifying their inputs.

    Construction checks alignment, timestamp dtype, and feature metadata.
    Training additionally validates chronological order, finite targets, and
    finite, nonnegative weights with positive total.
    """

    X: pl.DataFrame
    y: pl.Series
    time: pl.Series
    weight: pl.Series
    business_type: pl.Series | None = None
    cat_features: tuple[str, ...] = ()
    numerical_features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        lengths = {len(self.X), len(self.y), len(self.time), len(self.weight)}
        if self.business_type is not None:
            lengths.add(len(self.business_type))
        if lengths == {0}:
            raise ValueError("Pool must not be empty")
        if len(lengths) != 1:
            raise ValueError("Pool fields must have equal lengths")

        if self.time.dtype != pl.Date and self.time.dtype.base_type() != pl.Datetime:
            raise TypeError("Pool time must have Date or Datetime dtype")

        for label, names in (
            ("Categorical", self.cat_features),
            ("Numerical", self.numerical_features),
        ):
            if len(names) != len(set(names)):
                raise ValueError(f"{label} features must not contain duplicates")
            missing = sorted(set(names) - set(self.X.columns))
            if missing:
                raise ValueError(f"{label} features are missing from X: {missing}")
        overlap = sorted(set(self.cat_features) & set(self.numerical_features))
        if overlap:
            raise ValueError(f"Categorical and numerical features overlap: {overlap}")

    def __len__(self) -> int:
        """Return the number of aligned observations."""
        return len(self.X)

    def take(self, indices: Sequence[int] | np.ndarray) -> Pool:
        """Return aligned rows at positional indices, in the requested order."""
        index = np.asarray(indices, dtype=np.int64)
        return replace(
            self,
            X=self.X[index],
            y=self.y.gather(index),
            time=self.time.gather(index),
            weight=self.weight.gather(index),
            business_type=(
                None if self.business_type is None else self.business_type.gather(index)
            ),
        )

    def select_features(self, feature_names: Sequence[str]) -> Pool:
        """Return requested columns with both feature lists filtered to match."""
        names = list(feature_names)
        if not names:
            raise ValueError("At least one feature must be selected")
        missing = sorted(set(names) - set(self.X.columns))
        if missing:
            raise ValueError(f"Features are missing from X: {missing}")
        return replace(
            self,
            X=self.X.select(names),
            cat_features=tuple(name for name in self.cat_features if name in names),
            numerical_features=tuple(name for name in self.numerical_features if name in names),
        )

    def with_target(self, target: pl.Series | np.ndarray) -> Pool:
        """Replace the target; NumPy values retain the existing target name."""
        if isinstance(target, pl.Series):
            y = target
        else:
            values = np.asarray(target, dtype=float)
            if values.ndim != 1:
                raise ValueError("Target must contain exactly one value per row")
            y = pl.Series(self.y.name, values)
        if len(y) != len(self):
            raise ValueError("Target must contain exactly one value per row")
        return replace(self, y=y)

    def to_model_input(self, n_rows: int | None = None) -> pd.DataFrame:
        """Return all or the first n_rows as a categorical-safe pandas frame.

        Oversized limits return all rows. Null categorical values become
        ``"unknown"`` and categorical columns are cast to strings. Row order is
        preserved so serving examples stay aligned with timestamp metadata.
        """
        if n_rows is not None and n_rows < 0:
            raise ValueError("n_rows must be nonnegative")
        X = self.X if n_rows is None else self.X.head(n_rows)
        if self.cat_features:
            X = X.with_columns(
                pl.col(name).cast(pl.String).fill_null("unknown")
                for name in self.cat_features
            )
        return X.to_pandas()

    def to_catboost(self, baseline: float | np.ndarray | None = None) -> catboost.Pool:
        """Convert aligned data with an optional scalar or per-row baseline."""
        baseline_values = None
        if baseline is not None:
            if np.isscalar(baseline):
                baseline_values = np.full(len(self), float(baseline), dtype=float)
            else:
                baseline_values = np.asarray(baseline, dtype=float).reshape(-1)
                if baseline_values.shape != (len(self),):
                    raise ValueError("Baseline must contain exactly one value per row")
        time = self.time
        if time.dtype == pl.Date:
            time = time.cast(pl.Datetime("ms"))
        return catboost.Pool(
            data=self.to_model_input(),
            label=self.y.to_numpy(),
            cat_features=list(self.cat_features),
            feature_names=self.X.columns,
            timestamp=time.dt.timestamp(time_unit="ms").to_numpy(),
            weight=self.weight.to_numpy().reshape(-1),
            baseline=baseline_values,
        )
