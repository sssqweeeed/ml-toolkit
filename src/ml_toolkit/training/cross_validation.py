"""Time-series validation with explicit diagnostic and artifact policies."""

import logging
from collections.abc import Callable
from dataclasses import replace
from typing import Any, Literal

import mlflow
from ..tracking._model_state import audit_value
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from ..config import (
    CrossValidationConfig,
    DiagnosticsConfig,
    PostFitEvaluationConfig,
    TrackingConfig,
)
from ..models.base import Model
from .contracts import (
    PostFitEvaluator,
    TrainingData,
    _validate_post_fit_evaluator,
    validate_training_data,
)
from .evaluation import log_evaluation
from .results import CrossValidationResult

logger = logging.getLogger(__name__)


def _time_series_splits(n_rows: int, config: CrossValidationConfig):
    return enumerate(
        TimeSeriesSplit(
            n_splits=config.n_splits,
            gap=config.gap,
            test_size=config.test_size,
            max_train_size=config.max_train_size,
        ).split(np.arange(n_rows)),
        start=1,
    )


def _diagnostic_folds(
    config: CrossValidationConfig, diagnostics: DiagnosticsConfig | None
):
    evaluated = set(range(config.skip_splits + 1, config.n_splits + 1))
    if diagnostics is None:
        return set()
    if diagnostics.folds == "all":
        return evaluated
    if diagnostics.folds == "last":
        return {config.n_splits}
    if not set(diagnostics.folds) <= evaluated:
        raise ValueError("Diagnostic folds must be evaluated CV fold indices")
    return set(diagnostics.folds)


def cross_validate(
    *,
    data: TrainingData,
    params: dict[str, Any],
    model_factory: Callable[[dict[str, Any]], Model],
    metric: Callable,
    config: CrossValidationConfig,
    diagnostics: DiagnosticsConfig | None = None,
    tracking: TrackingConfig | None = None,
    post_fit_evaluator: PostFitEvaluator | None = None,
    post_fit: PostFitEvaluationConfig | None = None,
    stage: Literal["cv", "tuning"] = "cv",
) -> CrossValidationResult:
    """Evaluate chronological folds with task-specific optional diagnostics.

    Parameters
    ----------
    data : TrainingData
        Aligned, nonempty data with ascending timestamps and valid weights.
    params : dict[str, Any]
        Native model settings; each factory receives a copy.
    model_factory : Callable[[dict[str, Any]], Model]
        Factory producing a fresh adapter per evaluated fold.
    metric : Callable
        Receives target, predictions, and weights; returns numeric scores.
    config : CrossValidationConfig
        Row-based chronological splits and initial folds to skip.
    diagnostics : DiagnosticsConfig or None, default None
        Task-specific tables and figures for selected evaluated folds.
    tracking : TrackingConfig or None, default None
        Model packaging policy. None uses defaults, which do not package folds.
    post_fit_evaluator : PostFitEvaluator or None, default None
        Custom scalar/image evaluator; enabled calls require aligned object IDs.
    post_fit : PostFitEvaluationConfig or None, default None
        Stage, split, and image policy; None uses defaults.
    stage : {'cv', 'tuning'}, default 'cv'
        Current workflow stage for custom evaluation and packaging policy.

    Returns
    -------
    CrossValidationResult
        Validation scores and available zero-based best iteration indices.

    Notes
    -----
    Creates nested MLflow runs and logs mean metrics to the enclosing run.
    The caller configures MLflow and starts a parent run for standalone use.
    Model registration is reserved for final fitting.
    """
    if stage not in ("cv", "tuning"):
        raise ValueError("stage must be cv or tuning")
    tracking = TrackingConfig() if tracking is None else tracking
    post_fit = PostFitEvaluationConfig() if post_fit is None else post_fit
    evaluator = (
        post_fit_evaluator if stage in post_fit.stages and post_fit.splits else None
    )
    splits = _time_series_splits(len(data), config)
    validate_training_data(data)
    _validate_post_fit_evaluator(data, evaluator)
    diagnosed = _diagnostic_folds(config, diagnostics)
    logger.info(
        "Starting cross-validation with %s evaluated folds",
        config.n_splits - config.skip_splits,
    )
    fold_metrics: list[dict[str, float]] = []
    best_iterations: list[int] = []
    for fold_index, (train_index, validation_index) in splits:
        if fold_index <= config.skip_splits:
            logger.debug("Skipping fold %s", fold_index)
            continue
        train = data.take(train_index)
        validation = data.take(validation_index)
        logger.info("Starting fold %s", fold_index)
        logger.debug(
            "Fold %s: %s training rows, %s validation rows",
            fold_index,
            len(train),
            len(validation),
        )
        with mlflow.start_run(run_name=f"fold_{fold_index}", nested=True):
            mlflow.log_params(audit_value(params))
            model = model_factory(params.copy())
            model.fit(train, validation)
            scores = log_evaluation(
                model=model,
                train=train,
                validation=validation,
                train_prediction=model.predict_pool(train),
                validation_prediction=model.predict_pool(validation),
                metric=metric,
                diagnostics=diagnostics if fold_index in diagnosed else None,
                name=f"fold_{fold_index}",
                post_fit_evaluator=evaluator,
                post_fit=post_fit,
            )
            if (
                tracking.log_trial_models
                if stage == "tuning"
                else tracking.log_fold_models
            ):
                model.log(
                    validation, config=replace(tracking, registered_model_name=None)
                )
            if model.best_iteration is not None:
                best_iterations.append(int(model.best_iteration))
            fold_metrics.append(scores["val"])
        logger.info("Completed fold %s", fold_index)
    result = CrossValidationResult(tuple(fold_metrics), tuple(best_iterations))
    mlflow.log_metrics(
        {f"cv/{key}": value for key, value in result.mean_metrics.items()}
    )
    logger.info("Completed cross-validation with %s folds", len(fold_metrics))
    logger.debug("Cross-validation mean metrics: %s", result.mean_metrics)
    return result
