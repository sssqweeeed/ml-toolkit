"""Shared Polars data container for training and bundled model adapters."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Self

import catboost
import numpy as np
import pandas as pd
import polars as pl

from .types import FloatArray, IntArray

__all__ = [
    "Pool",
    "categorical_frame",
    "concat_pools",
    "row_indexer",
    "PoolSchema",
    "validate_compatible_pools",
    "validate_prediction",
    "validate_split_labels",
    "validate_training_categories",
    "validate_training_data",
]


@dataclass(frozen=True)
class Pool:
    """Hold aligned features, targets, timestamps, weights, and row metadata.

    Projects may supply categorical or continuous ``auxiliary_features`` for
    diagnostics. Only X is passed to models. Feature lists may describe a subset
    of X's columns.
    Fields cannot be reassigned; the contained frames and series are not deeply
    immutable. Pool methods return new objects without modifying their inputs.

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
    auxiliary_features : polars.DataFrame or None, default None
        Optional aligned diagnostic columns with names disjoint from X.
        Empty-width frames are normalized to None. Keyword-only.
    cat_features : tuple[str, ...], default ()
        Names of categorical features in X.
    numerical_features : tuple[str, ...], default ()
        Names of numerical features in X.
    object_id : polars.Series or None, default None
        Optional non-null string or integer IDs aligned with X. Keyword-only;
        repeated IDs are allowed. IDs retain
        their dtype and values through row and feature operations and are not
        added to model inputs.
    """

    X: pl.DataFrame
    y: pl.Series
    time: pl.Series
    weight: pl.Series
    auxiliary_features: pl.DataFrame | None = field(default=None, kw_only=True)
    cat_features: tuple[str, ...] = field(default=(), kw_only=True)
    numerical_features: tuple[str, ...] = field(default=(), kw_only=True)
    object_id: pl.Series | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        lengths = {len(self.X), len(self.y), len(self.time), len(self.weight)}
        if self.auxiliary_features is not None:
            if self.auxiliary_features.width == 0:
                object.__setattr__(self, "auxiliary_features", None)
            else:
                lengths.add(len(self.auxiliary_features))
                overlap = sorted(
                    set(self.X.columns) & set(self.auxiliary_features.columns)
                )
                if overlap:
                    raise ValueError(f"Auxiliary features overlap with X: {overlap}")
        if lengths == {0}:
            raise ValueError("Pool must not be empty")
        if len(lengths) != 1:
            raise ValueError("Pool fields must have equal lengths")
        if self.object_id is not None:
            validate_object_ids(self.object_id, len(self.X))

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

    def take(self, indices: Sequence[int] | IntArray) -> Self:
        """Return aligned rows at positional indices, in the requested order.

        Contiguous nonnegative positions share the source buffers. Reordered,
        repeated or noncontiguous positions gather rows into new buffers.

        Parameters
        ----------
        indices : sequence of int or IntArray
            Row positions. Repeated positions are preserved.

        Returns
        -------
        Pool
            New pool with every row field sliced together.

        Raises
        ------
        ValueError
            The requested selection is empty.
        """
        index = row_indexer(indices, len(self))
        return replace(
            self,
            X=self.X[index],
            y=self.y[index],
            time=self.time[index],
            weight=self.weight[index],
            object_id=None if self.object_id is None else self.object_id[index],
            auxiliary_features=(
                None
                if self.auxiliary_features is None
                else self.auxiliary_features[index]
            ),
        )

    def select_features(self, feature_names: Sequence[str]) -> Self:
        """Return requested columns with both feature lists filtered to match.

        Parameters
        ----------
        feature_names : sequence of str
            Nonempty selection of existing columns, in the desired order.

        Returns
        -------
        Pool
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

    def with_target(self, target: pl.Series | FloatArray) -> Self:
        """Return a pool with a replacement target and unchanged row metadata.

        Parameters
        ----------
        target : polars.Series or FloatArray of shape (n_rows,)
            Replacement values aligned with the existing rows. NumPy values
            retain the existing target name.

        Returns
        -------
        Pool
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

    def to_analysis_frame(self, columns: Sequence[str] | None = None) -> pl.DataFrame:
        """Return model and auxiliary columns for diagnostics, without joining rows.

        Parameters
        ----------
        columns : sequence of str or None, default None
            Unique requested names in output order. None includes X followed by
            all auxiliary columns. Targets, time, weights and IDs are not added.

        Returns
        -------
        polars.DataFrame
            Selected columns sharing source buffers. Inputs remain unchanged.

        Raises
        ------
        ValueError
            Requested names are duplicated or missing from both tables.
        """
        frame = self.X.clone()
        if self.auxiliary_features is not None:
            frame = frame.hstack(self.auxiliary_features)
        if columns is None:
            return frame
        names = list(columns)
        if len(names) != len(set(names)):
            raise ValueError("Analysis columns must contain unique names")
        missing = sorted(set(names) - set(frame.columns))
        if missing:
            raise ValueError(f"Features are missing from analysis data: {missing}")
        return frame.select(names)

    def to_model_input(
        self, n_rows: int | None = None, *, categorical_as_strings: bool = False
    ) -> pd.DataFrame:
        """Return all or the first n_rows as a categorical-safe pandas frame.

        Oversized limits return all rows. Null categorical values become
        ``"unknown"``. Categorical columns retain dictionary encoding with string
        labels. Row order is preserved so inputs stay aligned with metadata.

        Parameters
        ----------
        n_rows : int or None, default None
            Nonnegative row limit; None includes all rows.
        categorical_as_strings : bool, default False
            Expand categorical columns to strings for serialization or MLflow
            serving signatures. Model training and prediction accept the default
            pandas categorical columns without this expansion.

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
        if categorical_as_strings:
            frame = frame.with_columns(
                pl.col(name).cast(pl.String).fill_null("unknown")
                for name in self.cat_features
            )
            return frame.to_pandas(split_blocks=True)
        return categorical_frame(frame, self.cat_features)

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


@dataclass(frozen=True)
class PoolSchema:
    """Feature layout of a pool, without any of its rows.

    Attributes
    ----------
    features : polars.Schema
        Feature names, order and dtypes.
    cat_features, numerical_features : tuple[str, ...]
        Declared categorical and numerical features.
    auxiliary : polars.Schema or None
        Auxiliary feature names and dtypes, None without auxiliary features.
    """

    features: pl.Schema
    cat_features: tuple[str, ...]
    numerical_features: tuple[str, ...]
    auxiliary: pl.Schema | None

    @classmethod
    def of(cls, pool: Pool) -> Self:
        """Capture the feature layout of ``pool`` without referencing its data."""
        auxiliary = pool.auxiliary_features
        return cls(
            features=pool.X.schema,
            cat_features=pool.cat_features,
            numerical_features=pool.numerical_features,
            auxiliary=None if auxiliary is None else auxiliary.schema,
        )


def validate_compatible_pools(
    left: Pool | PoolSchema, right: Pool | PoolSchema, *, context: str
) -> None:
    """Require two pools to share features, feature metadata and auxiliary schema.

    Parameters
    ----------
    left, right : Pool or PoolSchema
        Pools, or layouts captured earlier with ``PoolSchema.of``, to compare.
    context : str
        Subject of the error message, for example ``"Split pools"``.

    Raises
    ------
    ValueError
        If feature schemas (names, order and dtypes) or categorical or
        numerical feature declarations differ, or, once those match, if
        auxiliary feature schemas differ.
    """
    a = left if isinstance(left, PoolSchema) else PoolSchema.of(left)
    b = right if isinstance(right, PoolSchema) else PoolSchema.of(right)
    if (
        a.features != b.features
        or a.cat_features != b.cat_features
        or a.numerical_features != b.numerical_features
    ):
        raise ValueError(f"{context} must have matching features and metadata")
    if a.auxiliary != b.auxiliary:
        raise ValueError(f"{context} must have matching auxiliary feature schemas")


def validate_object_ids(object_id: pl.Series, n_rows: int) -> None:
    if len(object_id) != n_rows:
        raise ValueError("object_id must have equal lengths with data rows")
    if object_id.dtype != pl.String and not object_id.dtype.is_integer():
        raise ValueError("object_id must have string or integer dtype")
    if object_id.null_count():
        raise ValueError("object_id must not contain null values")


def validate_training_data(data: Pool) -> None:
    """Check that a pool can be used for training.

    Parameters
    ----------
    data : Pool
        Pool to validate.

    Raises
    ------
    ValueError
        If lengths differ or are zero, auxiliary columns overlap with X, time is
        not an ascending non-null Date or Datetime, the target or weights are not
        finite, weights are negative or sum to zero, or object ids are invalid.
    """
    size = len(data)
    series = [data.y, data.time, data.weight]
    if data.auxiliary_features is not None:
        if len(data.auxiliary_features) != size:
            raise ValueError(
                "Auxiliary features must have equal lengths with data rows"
            )
        overlap = sorted(set(data.X.columns) & set(data.auxiliary_features.columns))
        if overlap:
            raise ValueError(f"Auxiliary features overlap with X: {overlap}")
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
    """Check categorical features exist and do not use the reserved value.

    Parameters
    ----------
    data : Pool
        Pool whose ``cat_features`` are checked.

    Raises
    ------
    ValueError
        If a categorical feature is missing from X or contains ``"unknown"``,
        which models use for missing categories.
    """
    for name in data.cat_features:
        if name not in data.X.columns:
            raise ValueError(f"Categorical feature is missing: {name}")
        column = data.X[name]
        if column.dtype.base_type() not in (pl.String, pl.Categorical, pl.Enum):
            column = column.unique().cast(pl.String)
        if (column == "unknown").any():
            raise ValueError(
                f"Categorical feature {name!r} contains reserved value 'unknown'"
            )


def row_indexer(indices: Sequence[int] | IntArray, n_rows: int) -> slice | IntArray:
    """Select rows by position, as a slice when the positions are contiguous.

    Parameters
    ----------
    indices : Sequence[int] or numpy.ndarray
        Row positions.
    n_rows : int
        Number of rows in the indexed data.

    Returns
    -------
    slice or numpy.ndarray
        A view-friendly slice for contiguous in-bounds ascending positions,
        otherwise the positions as an int64 array in the given order.
    """
    if (
        isinstance(indices, range)
        and indices.step == 1
        and 0 <= indices.start < indices.stop <= n_rows
    ):
        return slice(indices.start, indices.stop)
    index = np.asarray(indices, dtype=np.int64)
    if (
        index.ndim == 1
        and index.size
        and 0 <= index[0]
        and index[-1] < n_rows
        and index[-1] - index[0] == index.size - 1
        and np.all(index[1:] > index[:-1])
    ):
        return slice(int(index[0]), int(index[-1]) + 1)
    return index


def categorical_frame(data: pl.DataFrame, categorical: Sequence[str]) -> pd.DataFrame:
    """Convert Polars features to pandas with categorical columns encoded.

    Training and every prediction path use this single conversion.

    Parameters
    ----------
    data : polars.DataFrame
        Feature columns.
    categorical : Sequence[str]
        Names of columns to cast to categorical; nulls become ``"unknown"``.

    Returns
    -------
    pandas.DataFrame
        Features with categorical dtype for the listed columns.
    """
    expressions: list[pl.Expr] = []
    for name in categorical:
        column = pl.col(name)
        dtype = data.schema[name]
        if dtype == pl.Enum:
            column = column.cast(pl.Categorical)
        elif dtype != pl.Categorical:
            column = column.cast(pl.String).cast(pl.Categorical)
        expressions.append(column.fill_null("unknown"))
    # Separate numeric blocks let Arrow share the original numeric buffers.
    return data.with_columns(expressions).to_pandas(split_blocks=True)


def validate_split_labels(data: Pool, split: pl.Series) -> None:
    """Check split labels are aligned with a pool and name one held-out split.

    Parameters
    ----------
    data : Pool
        Pool the labels describe.
    split : polars.Series
        One non-null String label per row: ``"train"``, ``"val"`` or ``"test"``.

    Raises
    ------
    ValueError
        If labels are misaligned, null, not String, unknown, or contain both
        ``"val"`` and ``"test"``.
    """
    if (
        len(split) != len(data)
        or split.dtype != pl.String
        or split.null_count()
        or not split.is_in(("train", "val", "test")).all()
    ):
        raise ValueError(
            "split must contain aligned non-null String train/val/test labels"
        )
    if {"val", "test"} <= set(split.unique()):
        raise ValueError(
            "Reporting accepts only one held-out split: val or test, not both"
        )


def concat_pools(train: Pool, heldout: Pool | None) -> Pool:
    """Stack a training pool and an optional held-out pool.

    Parameters
    ----------
    train : Pool
        Leading rows.
    heldout : Pool or None
        Rows appended after ``train``; None returns ``train`` unchanged.

    Returns
    -------
    Pool
        Pool with the rows of both inputs.

    Raises
    ------
    ValueError
        If feature schemas, feature metadata, auxiliary schemas or optional
        fields differ.
    """
    if heldout is None:
        return train
    validate_compatible_pools(train, heldout, context="Reporting pools")

    def optional(left: pl.Series | None, right: pl.Series | None) -> pl.Series | None:
        if left is None and right is None:
            return None
        if left is None or right is None:
            raise ValueError("Reporting pools must have matching optional fields")
        return pl.concat([left, right], rechunk=False)

    return Pool(
        X=pl.concat([train.X, heldout.X], rechunk=False),
        y=pl.concat([train.y, heldout.y], rechunk=False),
        time=pl.concat([train.time, heldout.time], rechunk=False),
        weight=pl.concat([train.weight, heldout.weight], rechunk=False),
        auxiliary_features=(
            None
            if train.auxiliary_features is None or heldout.auxiliary_features is None
            else pl.concat(
                [train.auxiliary_features, heldout.auxiliary_features], rechunk=False
            )
        ),
        object_id=optional(train.object_id, heldout.object_id),
        cat_features=train.cat_features,
        numerical_features=train.numerical_features,
    )


def validate_prediction(data: Pool, prediction: FloatArray) -> None:
    """Check there is one finite prediction per pool row.

    Parameters
    ----------
    data : Pool
        Pool the prediction describes.
    prediction : numpy.ndarray
        One-dimensional predictions.

    Raises
    ------
    ValueError
        If the shape is not ``(len(data),)`` or any value is not finite.
    """
    if prediction.shape != (len(data),) or not np.isfinite(prediction).all():
        raise ValueError("Predictions must contain one finite value per row")
