"""Time-series validation with MLflow logging per fold."""

import logging
from collections.abc import Callable
from typing import Any

import mlflow
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from .contracts import (
    PostFitEvaluator, TrainingData, _validate_post_fit_evaluator, validate_training_data,
)
from .evaluation import DiagnosticsOptions, log_evaluation
from .results import CrossValidationResult

logger = logging.getLogger(__name__)


def _time_series_splits(n_rows: int, n_splits: int, skip_splits: int):
    """Return the folds and validate the shared chronological split settings."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if not 0 <= skip_splits < n_splits:
        raise ValueError("skip_splits must be between 0 and n_splits - 1")
    return enumerate(TimeSeriesSplit(n_splits=n_splits).split(np.arange(n_rows)), start=1)


def cross_validate(
    *, data: TrainingData, params: dict[str, Any], model_factory: Callable,
    metric: Callable, n_splits: int, skip_splits: int = 0,
    diagnostics: DiagnosticsOptions | None = None,
    post_fit_evaluator: PostFitEvaluator | None = None,
) -> CrossValidationResult:
    """Evaluate a model on expanding chronological folds and log each fold.

    Parameters
    ----------
    data : TrainingData
        Project-owned data with equal, nonzero row counts; finite targets;
        non-null ascending timestamps; and finite, nonnegative weights with
        positive total.
    params : dict[str, Any]
        Parameters passed to each new model.
    model_factory : Callable
        Factory receiving a parameter dictionary and returning a model
        implementing the Model protocol.
    metric : Callable
        Callable receiving target, predictions, and weights as NumPy arrays;
        return a dataclass or mapping of names to numeric scalar values.
    n_splits : int
        Number of time-series folds; at least two.
    skip_splits : int, default 0
        Initial folds to omit, from zero to n_splits - 1.
    diagnostics : DiagnosticsOptions or None, default None
        Optional per-fold PSI and plot artifacts.
    post_fit_evaluator : PostFitEvaluator or None, default None
        User-owned evaluator receiving aligned IDs, outputs, targets, and split
        after each fold fit. Requires data.object_id. The same instance is reused
        without resetting; results are logged as train/custom/* and val/custom/*
        in fold runs and excluded from CV aggregation and optimization.
        Optional write_artifacts returns images to upload, or None to skip.

    Returns
    -------
    CrossValidationResult
        Validation metric dictionaries and available zero-based best iteration
        indices.

    Raises
    ------
    ValueError
        Input data, chronological split settings, post-fit IDs, predictions,
        metric names, or finite metric values are invalid.
    TypeError
        The evaluator has no evaluate method, IDs have an unsupported type, or
        its result is not a dictionary of floating scalars.

    Notes
    -----
    Each fold creates a nested MLflow run, fits a fresh model, evaluates both
    splits, and logs the model using validation rows as the serving example.
    Mean validation metrics are logged to the enclosing run under "cv/".
    Configure MLflow and start a parent run when calling this independently.
    """
    splits = _time_series_splits(len(data), n_splits, skip_splits)
    validate_training_data(data)
    _validate_post_fit_evaluator(data, post_fit_evaluator)
    logger.info("Starting cross-validation with %s evaluated folds", n_splits - skip_splits)
    fold_metrics: list[dict[str, float]] = []
    best_iterations: list[int] = []
    for fold_index, (train_index, validation_index) in splits:
        if fold_index <= skip_splits:
            logger.debug("Skipping fold %s", fold_index)
            continue
        train = data.take(train_index)
        validation = data.take(validation_index)
        _validate_post_fit_evaluator(train, post_fit_evaluator)
        _validate_post_fit_evaluator(validation, post_fit_evaluator)
        logger.info("Starting fold %s", fold_index)
        logger.debug(
            "Fold %s: %s training rows, %s validation rows",
            fold_index,
            len(train),
            len(validation),
        )
        with mlflow.start_run(run_name=f"fold_{fold_index}", nested=True):
            mlflow.log_params(params)
            model = model_factory(params)
            model.fit(train, validation)
            scores = log_evaluation(
                model=model, train=train, validation=validation,
                train_prediction=model.predict_pool(train),
                validation_prediction=model.predict_pool(validation),
                metric=metric, diagnostics=diagnostics, name=f"fold_{fold_index}",
                post_fit_evaluator=post_fit_evaluator,
            )
            model.log(validation)
            if model.best_iteration is not None:
                best_iterations.append(int(model.best_iteration))
            fold_metrics.append(scores["val"])
        logger.info("Completed fold %s", fold_index)
    if not fold_metrics:
        raise ValueError("No folds were processed")
    result = CrossValidationResult(tuple(fold_metrics), tuple(best_iterations))
    mlflow.log_metrics({f"cv/{key}": value for key, value in result.mean_metrics.items()})
    logger.info("Completed cross-validation with %s folds", len(fold_metrics))
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Cross-validation mean metrics: %s", result.mean_metrics)
    return result
