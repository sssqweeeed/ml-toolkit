"""Example project-owned Polars data container implementing ml_toolkit.Pool.

Copy and adapt this implementation in your project. The toolkit consumes its
structural interface; this example is not part of the installed package.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import catboost
import numpy as np
import pandas as pd
import polars as pl

from ml_toolkit.types import FloatArray, IntArray


@dataclass(frozen=True)
class ProjectPool:
    """Hold aligned features, targets, timestamps, weights, and row metadata.

    Projects prepare the data and may optionally supply ``business_type`` for
    grouped plots. Feature lists may describe only a subset of X's columns.
    Fields cannot be reassigned; the contained frames and series are not deeply
    immutable. ProjectPool methods return new objects without modifying their
    inputs.

    Construction checks alignment, timestamp dtype, and feature metadata.
    Training additionally validates chronological order, finite targets, and
    finite, nonnegative weights with positive total.

    Parameters
    ----------
    X : polars.DataFrame
        Feature columns, with one row per observation.
    y : polars.Series
        Target values aligned with X.
    time : polars.Series
        Date or Datetime values aligned with X.
    weight : polars.Series
        Row weights aligned with X.
    business_type : polars.Series or None, default None
        Optional aligned groups for diagnostic plots.
    cat_features : tuple[str, ...], default ()
        Names of categorical features in X.
    numerical_features : tuple[str, ...], default ()
        Names of numerical features in X.
    object_id : polars.Series or None, default None
        Optional non-null string or integer IDs aligned with X. Keyword-only;
        repeated IDs are allowed. Required for post-fit evaluation. IDs retain
        their dtype and values through row and feature operations and are not
        added to model inputs.
    """

    X: pl.DataFrame
    y: pl.Series
    time: pl.Series
    weight: pl.Series
    business_type: pl.Series | None = None
    cat_features: tuple[str, ...] = ()
    numerical_features: tuple[str, ...] = ()
    object_id: pl.Series | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        lengths = {len(self.X), len(self.y), len(self.time), len(self.weight)}
        if self.business_type is not None:
            lengths.add(len(self.business_type))
        if lengths == {0}:
            raise ValueError("ProjectPool must not be empty")
        if len(lengths) != 1:
            raise ValueError("ProjectPool fields must have equal lengths")
        if self.object_id is not None:
            if len(self.object_id) != len(self.X):
                raise ValueError("object_id must have equal lengths with data rows")
            if (
                self.object_id.dtype != pl.String
                and not self.object_id.dtype.is_integer()
            ):
                raise ValueError("object_id must have string or integer dtype")
            if self.object_id.null_count():
                raise ValueError("object_id must not contain null values")

        if self.time.dtype != pl.Date and self.time.dtype.base_type() != pl.Datetime:
            raise TypeError("ProjectPool time must have Date or Datetime dtype")

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

    def take(self, indices: Sequence[int] | IntArray) -> ProjectPool:
        """Return aligned rows at positional indices, in the requested order.

        Parameters
        ----------
        indices : sequence of int or IntArray
            Row positions. Repeated positions are preserved.

        Returns
        -------
        ProjectPool
            New pool with every row field sliced together.

        Raises
        ------
        ValueError
            The requested selection is empty.
        """
        index = np.asarray(indices, dtype=np.int64)
        return replace(
            self,
            X=self.X[index],
            y=self.y.gather(index),
            time=self.time.gather(index),
            weight=self.weight.gather(index),
            object_id=None if self.object_id is None else self.object_id.gather(index),
            business_type=(
                None if self.business_type is None else self.business_type.gather(index)
            ),
        )

    def select_features(self, feature_names: Sequence[str]) -> ProjectPool:
        """Return requested columns with both feature lists filtered to match.

        Parameters
        ----------
        feature_names : sequence of str
            Nonempty selection of existing columns, in the desired order.

        Returns
        -------
        ProjectPool
            New pool retaining all row metadata and matching feature lists.

        Raises
        ------
        ValueError
            The selection is empty or a requested feature is missing.
        """
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
            numerical_features=tuple(
                name for name in self.numerical_features if name in names
            ),
        )

    def with_target(self, target: pl.Series | FloatArray) -> ProjectPool:
        """Return a pool with a replacement target and unchanged row metadata.

        Parameters
        ----------
        target : polars.Series or FloatArray of shape (n_rows,)
            Replacement values aligned with the existing rows. NumPy values
            retain the existing target name.

        Returns
        -------
        ProjectPool
            New pool with the replacement target.

        Raises
        ------
        ValueError
            The target does not contain exactly one value per row.
        """
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

        Parameters
        ----------
        n_rows : int or None, default None
            Nonnegative row limit; None includes all rows.

        Returns
        -------
        pandas.DataFrame
            Feature columns in their existing order, without row metadata.

        Raises
        ------
        ValueError
            n_rows is negative.
        """
        if n_rows is not None and n_rows < 0:
            raise ValueError("n_rows must be nonnegative")
        frame = self.X if n_rows is None else self.X.head(n_rows)
        if self.cat_features:
            frame = frame.with_columns(
                pl.col(name).cast(pl.String).fill_null("unknown")
                for name in self.cat_features
            )
        return frame.to_pandas()

    def to_catboost(self, baseline: float | FloatArray | None = None) -> catboost.Pool:
        """Convert aligned data with an optional scalar or per-row baseline.

        Parameters
        ----------
        baseline : float, FloatArray of shape (n_rows,), or None, default None
            Initial raw model score for each row. A scalar is broadcast; None
            lets CatBoost choose its normal initialization.

        Returns
        -------
        catboost.Pool
            Native pool with features, targets, weights, categorical feature
            names, millisecond timestamps, and the optional baseline.

        Raises
        ------
        ValueError
            A per-row baseline has the wrong number of values.
        """
        baseline_values = None
        if baseline is not None:
            if isinstance(baseline, float):
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
