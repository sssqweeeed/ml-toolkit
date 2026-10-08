"""Shared interface for toolkit Pool and compatible project data containers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, Self

import catboost
import pandas as pd
import polars as pl

from ..types import FloatArray, IntArray

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
            Supplied validation rows with aligned features and weights.

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
        evaluates test only after final fitting when test_size is configured.
    """

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class PostFitEvaluator(ABC):
    """Describe a user-owned evaluator called after each evaluated model fit.

    Notes
    -----
    Initialization and helper methods are unrestricted; the default image writer does
    nothing. The same instance is reused for splits, tuning trials, post-tuning
    evaluation, and final fitting, without copying or resetting its state.
    Reference data and caches may be stored internally, but returned metrics
    should describe the supplied rows rather than accumulate unrelated fits.
    Intermediate feature-selection fits do not invoke this evaluator.
    Subclass this evaluator and implement evaluate. Override write_artifacts
    to save images; its default implementation returns an empty list.
    """

    @abstractmethod
    def evaluate(
        self,
        *,
        object_id: pl.Series,
        y_pred: FloatArray,
        y_true: FloatArray,
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

    def write_artifacts(
        self,
        *,
        object_id: pl.Series,
        y_pred: FloatArray,
        y_true: FloatArray,
        split: DatasetSplit,
        artifact_dir: Path,
    ) -> list[Path]:
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
        list[pathlib.Path]
            Existing image files to upload under custom/<split>/ in the current
            MLflow run. Paths may be absolute or relative to artifact_dir. An empty list skips upload. Only returned files are logged; PNG,
            JPEG, SVG, WebP, GIF, TIFF, and BMP extensions are accepted. Tables
            are not accepted. The directory is removed after the call and
            uploads, including on errors; returned paths must not be retained.

        Notes
        -----
        Invoked wherever scalar evaluation runs, including Optuna trials and
        final fitting, subject to PostFitEvaluationConfig. Choose whether to
        render inside this method. Inputs must not be mutated. Errors propagate
        with evaluation stage and split context.
        """
        return []


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
    numerical_features : tuple[str, ...]
        Declared numerical columns; may describe a subset of X.
    object_id : polars.Series or None
        Optional aligned IDs for custom evaluation.
    business_type : polars.Series or None
        Optional aligned groups for monthly diagnostics.

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

    @property
    def numerical_features(self) -> tuple[str, ...]:
        """Return declared numerical feature names."""
        ...

    @property
    def object_id(self) -> pl.Series | None:
        """Return aligned IDs, or None when custom evaluation is unavailable."""
        ...

    @property
    def business_type(self) -> pl.Series | None:
        """Return aligned diagnostic groups, or None."""
        ...

    def __len__(self) -> int:
        """Return the number of aligned observations."""
        ...

    def take(self, indices: Sequence[int] | IntArray) -> Self:
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

    def with_target(self, target: pl.Series | FloatArray) -> Self:
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

    def to_catboost(self, baseline: float | FloatArray | None = None) -> catboost.Pool:
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


TrainingStage = Literal["tuning", "validation", "final"]
