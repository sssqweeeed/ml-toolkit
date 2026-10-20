"""Shared data contract and interfaces for training extensions."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

import polars as pl

from ..data import Pool as Pool
from ..types import FloatArray

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

    def __call__(self, model: "Model", data: "Pool") -> Mapping[str, float]:
        """Return finite ranking values keyed by every current feature name.

        Parameters
        ----------
        model : Model
            Fitted adapter for the current feature subset.
        data : Pool
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

    Parameters
    ----------
    name : str
        Required namespace for metrics and artifacts. Start with an ASCII letter
        or digit; remaining characters may also be underscores, hyphens or dots.
        Names must be unique without regard to case within one training call.

    Attributes
    ----------
    name : str
        Read-only namespace, preserved across calls.

    Notes
    -----
    Subclasses with a constructor must call ``super().__init__(name=name)``.
    The default artifact writer does nothing. The same instance is reused for
    splits, tuning trials, post-tuning evaluation, and final fitting, without
    copying or resetting its state.
    Reference data and caches may be stored internally, but returned metrics
    should describe the supplied rows rather than accumulate unrelated fits.
    Intermediate feature-selection fits do not invoke this evaluator.
    Subclass this evaluator and implement evaluate. Override write_artifacts
    to save files; its default implementation returns an empty list.
    """

    def __init__(self, *, name: str) -> None:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) is None:
            raise ValueError(
                "Evaluator name must start with an ASCII letter or digit and "
                "contain only ASCII letters, digits, underscores, hyphens or dots"
            )
        self._name = name

    @property
    def name(self) -> str:
        """Return the read-only namespace used for this evaluator's results."""
        return self._name

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
            ``<split>/<name>/`` prefix and logs to the current MLflow run. An
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
        """Save and select artifacts to log for the current split.

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
            Fresh temporary directory owned by the toolkit. Save files inside
            this directory; nested directories are preserved in MLflow.

        Returns
        -------
        list[pathlib.Path]
            Existing files to upload under ``<name>/<split>/`` in the
            current MLflow run. Paths may be absolute or relative to artifact_dir.
            An empty list skips upload. Only returned files are logged; PNG,
            JPEG, SVG, WebP, GIF, TIFF, BMP, CSV, Parquet, JSON, and HTML extensions are accepted. The directory is removed after the call and
            uploads, including on errors; returned paths must not be retained.

        Notes
        -----
        Invoked wherever scalar evaluation runs, including Optuna trials and
        final fitting, subject to PostFitEvaluationConfig. Choose whether to
        render inside this method. Inputs must not be mutated. Errors propagate
        with evaluator name, evaluation stage and split context.
        """
        return []


TrainingStage = Literal["tuning", "validation", "final"]
