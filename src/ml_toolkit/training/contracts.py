"""Shared interface for toolkit Pool and compatible project data containers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Self

import numpy as np
import pandas as pd
import polars as pl

from .._validation import _validate_object_id

if TYPE_CHECKING:
    from ..models.base import Model


class FeatureRanker(Protocol):
    """Describe a callable ranking a fitted adapter's current input columns.

    Notes
    -----
    Higher values indicate greater importance. Implementations may inspect the
    adapter's native_model, but must preserve its state and all input metadata.
    Explicit inheritance from this protocol is unnecessary.
    """

    def __call__(self, model: "Model", data: "TrainingData") -> Mapping[str, float]:
        """Return finite ranking values keyed by every current feature name.

        Parameters
        ----------
        model : Model
            Fitted adapter for the current feature subset.
        data : TrainingData
            Internal validation rows with aligned features and weights.

        Returns
        -------
        Mapping[str, float]
            Exactly one finite numeric importance per data.X column. Signed
            values are allowed; larger values rank higher.
        """
        ...


class DatasetSplit(StrEnum):
    """Identify an evaluated dataset split.

    Attributes
    ----------
    TRAIN : DatasetSplit
        Rows used to fit the model; serialized as "train".
    VAL : DatasetSplit
        Validation rows; serialized as "val".
    TEST : DatasetSplit
        Independent test rows; serialized as "test". The training workflow
        currently evaluates train and val only.
    """

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class PostFitEvaluator(Protocol):
    """Describe a user-owned evaluator called after each evaluated model fit.

    Notes
    -----
    Initialization and helper methods are unrestricted; explicit inheritance is
    unnecessary. The same instance is reused for folds, tuning trials, repeated
    validation, and final fitting, without copying or resetting its state.
    Reference data and caches may be stored internally, but returned metrics
    should describe the supplied rows rather than accumulate unrelated fits.
    Intermediate feature-selection fits do not invoke this evaluator.
    Implement PostFitArtifactWriter.write_artifacts on the same object to
    optionally save additional images after each evaluation. No training flag
    is required; returning None from that method skips image logging.
    """

    def evaluate(
        self,
        *,
        object_id: pl.Series,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        split: DatasetSplit,
    ) -> dict[str, float]:
        """Calculate additional scalar metrics for one complete dataset split.

        Parameters
        ----------
        object_id : polars.Series of shape (n_samples,)
            Non-null string or integer IDs in prediction order; repeats allowed.
        y_pred : numpy.ndarray of shape (n_samples,)
            Finite model outputs. Binary classifiers supply positive-class
            probabilities; regressors supply predicted target values.
        y_true : numpy.ndarray of shape (n_samples,)
            Finite observed target values aligned with object_id and y_pred.
        split : DatasetSplit
            Strictly train, val, or test, passed as an enum member.

        Returns
        -------
        dict[str, float]
            Finite Python or NumPy floating scalars keyed by nonempty metric
            names. Names may contain word characters, spaces, periods, dashes,
            and slashes, but no relative path segments. The toolkit adds the
            "<split>/custom/" prefix and logs to the current MLflow run. An
            empty dictionary skips logging. Results are excluded from ordinary
            CV metrics and optimization. Inputs must not be mutated.
        """
        ...


class PostFitArtifactWriter(Protocol):
    """Describe optional image generation on a post-fit evaluator.

    Notes
    -----
    A user-owned evaluator may implement both evaluate and write_artifacts
    without inheriting either protocol. Image generation receives the same
    inputs immediately after scalar evaluation, even when metrics are empty.
    Each method can independently prepare its data; no caching is required.
    """

    def write_artifacts(
        self,
        *,
        object_id: pl.Series,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        split: DatasetSplit,
        artifact_dir: Path,
    ) -> list[Path] | None:
        """Save and select images to log for the current split.

        Parameters
        ----------
        object_id : polars.Series of shape (n_samples,)
            Aligned non-null string or integer IDs; repeats are allowed.
        y_pred : numpy.ndarray of shape (n_samples,)
            Finite task-appropriate model outputs, as supplied to evaluate.
        y_true : numpy.ndarray of shape (n_samples,)
            Finite observed target values aligned with IDs and outputs.
        split : DatasetSplit
            The current train, val, or test split.
        artifact_dir : pathlib.Path
            Fresh temporary directory owned by the toolkit. Save images inside
            this directory; nested directories are preserved in MLflow.

        Returns
        -------
        list[pathlib.Path] or None
            Existing image files to upload under custom/<split>/ in the current
            MLflow run. Paths may be absolute or relative to artifact_dir. None
            or an empty list skips upload. Only returned files are logged; PNG,
            JPEG, SVG, WebP, GIF, TIFF, and BMP extensions are accepted. Tables
            are not accepted. The directory is removed after the call and
            uploads, including on errors; returned paths must not be retained.

        Notes
        -----
        Invoked wherever scalar evaluation runs, including Optuna trials and
        final fitting, regardless of DiagnosticsOptions. Choose whether to
        render inside this method. Inputs must not be mutated. Errors propagate
        with evaluation stage and split context.
        """
        ...


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


class IdentifiedTrainingData(TrainingData, Protocol):
    """Extend training data with optional aligned object identifiers.

    Attributes
    ----------
    object_id : polars.Series or None
        Non-null string or integer IDs with one value per row. IDs are required
        when a post-fit evaluator is supplied; repeated IDs are allowed. take
        must slice them in row order; select_features and with_target preserve
        them. IDs are metadata and are not automatically included in X.
    """

    @property
    def object_id(self) -> pl.Series | None:
        """Return aligned IDs, or None when identifiers were not supplied."""
        ...


def _validate_post_fit_evaluator(
    data: TrainingData, evaluator: PostFitEvaluator | None,
) -> None:
    if evaluator is None:
        return
    if not callable(getattr(evaluator, "evaluate", None)):
        raise TypeError("post_fit_evaluator must provide an evaluate method")
    writer = getattr(evaluator, "write_artifacts", None)
    if writer is not None and not callable(writer):
        raise TypeError("post_fit_evaluator.write_artifacts must be callable")
    object_id = getattr(data, "object_id", None)
    if object_id is None:
        raise ValueError("object_id is required when post_fit_evaluator is supplied")
    _validate_object_id(object_id, len(data))


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
