"""Compose model-owned selection, tuning, validation, and final fitting."""

import logging
from collections.abc import Callable
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import optuna

from ..config import PUDiagnosticsConfig, TrainingConfig
from ..metrics.pu import PUMetric
from ..models.base import Model
from .contracts import (
    DatasetSplit,
    PostFitEvaluator,
    TrainingData,
    _validate_post_fit_evaluator,
    validate_training_data,
)
from .cross_validation import _time_series_splits, cross_validate
from .evaluation import _diagnostics_requested, _log_post_fit_evaluation
from .feature_selection import feature_elimination
from .results import TrainingResult
from .tuning import tune_optuna

logger = logging.getLogger(__name__)


def _configuration_json(value):
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "type": type(value).__name__,
            **{
                item.name: _configuration_json(getattr(value, item.name))
                for item in fields(value)
            },
        }
    if isinstance(value, tuple):
        return [_configuration_json(item) for item in value]
    if isinstance(value, (Path, date, datetime)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def train_model(
    *,
    data: TrainingData,
    params: dict[str, Any],
    model_factory: Callable[[dict[str, Any]], Model],
    metric: Callable,
    run_name: str,
    config: TrainingConfig,
    param_space: Callable[[optuna.Trial], dict[str, Any]] | None = None,
    post_fit_evaluator: PostFitEvaluator | None = None,
) -> TrainingResult:
    """Select, tune, validate, and fit using an immutable workflow configuration.

    Parameters
    ----------
    data : TrainingData
        Aligned nonempty project data with finite targets, ascending non-null
        timestamps, and finite nonnegative weights with positive total.
    params : dict[str, Any]
        Native model settings. Factories receive copies; input is not mutated.
    model_factory : Callable[[dict[str, Any]], Model]
        Factory whose adapters implement fit, predict_pool, select_features,
        best_iteration, and log(sample, config=TrackingConfig).
    metric : Callable
        Callable receiving target, predictions, and weights as NumPy arrays;
        returns a dataclass or mapping of metric names to scalar values.
    run_name : str
        Name of the enclosing MLflow run.
    config : TrainingConfig
        Selection, tuning, chronological CV, diagnostics, final iteration, model
        packaging, and custom evaluator policies.
    param_space : Callable[[optuna.Trial], dict[str, Any]] or None, default None
        Trial parameter overrides; required when config.tuning is enabled.
    post_fit_evaluator : PostFitEvaluator or None, default None
        Custom evaluator receiving aligned IDs, predictions, targets, and split.
        Enabled calls require data.object_id. The instance is reused without
        resetting. Optional images follow config.post_fit.write_artifacts.

    Returns
    -------
    TrainingResult
        Final fitted model, CV scores, effective native settings, and features.

    Raises
    ------
    ValueError
        Data, split sizes, feature count, tuning dependencies, or PU task settings
        are invalid. Enabled custom evaluation lacks valid aligned IDs.

    Notes
    -----
    The caller owns the MLflow URI and experiment. Creates a nested run when a
    parent is active. Selection uses training rows of the first evaluated fold
    only. Tuning uses the same CV splits. Winning parameters are evaluated again
    only when evaluation diagnostics, fold packaging, or custom CV evaluation
    need a dedicated pass. Configuration and resolved settings are saved as
    JSON artifacts. Final fitting uses all rows, with early stopping disabled.

    Examples
    --------
    >>> from ml_toolkit.config import CrossValidationConfig, TrainingConfig
    >>> settings = TrainingConfig(cv=CrossValidationConfig(n_splits=3))
    >>> result = train_model(  # doctest: +SKIP
    ...     data=data, params=params, model_factory=model_factory,
    ...     metric=metric, run_name="experiment", config=settings,
    ... )
    """
    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be a TrainingConfig")
    validate_training_data(data)
    splits = list(_time_series_splits(len(data), config.cv))
    if config.tuning is not None and param_space is None:
        raise ValueError("param_space is required when tuning is enabled")
    if config.feature_selection is not None:
        if config.feature_selection.n_features > len(data.X.columns):
            raise ValueError("n_features must not exceed the input feature count")
        first_train = next(
            indices for index, (indices, _) in splits if index > config.cv.skip_splits
        )
        if len(first_train) < 2:
            raise ValueError(
                "First evaluated fold has fewer than two training rows for feature selection"
            )
        if params.get("loss", params.get("loss_function")) in (
            "uPU",
            "nnPU",
        ) or isinstance(metric, PUMetric):
            raise NotImplementedError(
                "CatBoost's built-in feature selection cannot optimize PU risk"
            )
    if (
        isinstance(metric, PUMetric)
        and config.diagnostics is not None
        and not isinstance(config.diagnostics, PUDiagnosticsConfig)
    ):
        raise ValueError("PU models require PUDiagnosticsConfig")
    active_stages = {"cv"}
    if config.tuning is not None:
        active_stages.add("tuning")
    enabled_evaluator = bool(
        set(config.post_fit.stages) & active_stages and config.post_fit.splits
    ) or ("final" in config.post_fit.stages and "train" in config.post_fit.splits)
    _validate_post_fit_evaluator(
        data, post_fit_evaluator if enabled_evaluator else None
    )
    logger.info("Starting training run %s", run_name)
    logger.debug("Training data: %s rows, %s features", len(data), len(data.X.columns))
    with mlflow.start_run(run_name=run_name, nested=mlflow.active_run() is not None):
        mlflow.log_dict(_configuration_json(config), "configuration/requested.json")
        working = data
        if config.feature_selection is not None:
            selected = feature_elimination(
                data=data.take(first_train),
                params=params,
                model_factory=model_factory,
                config=config.feature_selection,
            )
            working = data.select_features(selected.X.columns)
            mlflow.log_dict({"features": working.X.columns}, "selection/features.json")
        final_params = params.copy()
        if config.tuning is not None:
            tuning = tune_optuna(
                data=working,
                base_params=final_params,
                model_factory=model_factory,
                metric=metric,
                param_space=param_space,
                config=config.tuning,
                cv=config.cv,
                tracking=config.tracking,
                post_fit_evaluator=post_fit_evaluator,
                post_fit=config.post_fit,
            )
            final_params.update(tuning.best_params)
            cv = tuning.cross_validation
            need_cv = (
                _diagnostics_requested(config.diagnostics)
                or config.tracking.log_fold_models
                or (
                    post_fit_evaluator is not None
                    and "cv" in config.post_fit.stages
                    and config.post_fit.splits
                )
            )
        else:
            need_cv = True
        if need_cv:
            cv = cross_validate(
                data=working,
                params=final_params,
                model_factory=model_factory,
                metric=metric,
                config=config.cv,
                diagnostics=config.diagnostics,
                tracking=config.tracking,
                post_fit_evaluator=post_fit_evaluator,
                post_fit=config.post_fit,
            )
        selected_iterations = (
            cv.final_iterations
            if config.final_fit.iteration_policy == "cv_median"
            else None
        )
        if selected_iterations is not None:
            for name in ("iterations", "n_estimators", "num_trees"):
                if name in final_params:
                    final_params[name] = selected_iterations
                    break
        final_params.pop("early_stopping_rounds", None)
        if "use_best_model" in final_params:
            final_params["use_best_model"] = False
        mlflow.log_dict(
            {
                "features": working.X.columns,
                "model_params": final_params,
                "cv_selected_iterations": selected_iterations,
            },
            "configuration/resolved.json",
        )
        with mlflow.start_run(run_name="final_fit", nested=True):
            logger.info("Starting final fit for run %s", run_name)
            logger.debug("Final fit selected iterations: %s", selected_iterations)
            mlflow.log_params(final_params)
            if selected_iterations is not None:
                mlflow.log_param("selected_iterations", selected_iterations)
            model = model_factory(final_params.copy())
            model.fit(working)
            if (
                post_fit_evaluator is not None
                and "final" in config.post_fit.stages
                and "train" in config.post_fit.splits
            ):
                _log_post_fit_evaluation(
                    evaluator=post_fit_evaluator,
                    data=working,
                    prediction=model.predict_pool(working),
                    split=DatasetSplit.TRAIN,
                    name="final_fit",
                    write_artifacts=config.post_fit.write_artifacts,
                )
            if config.tracking.log_final_model:
                model.log(working, config=config.tracking)
            mlflow.log_params(
                {
                    "train_n_rows": len(working),
                    "train_date_start": str(working.time.min()),
                    "train_date_end": str(working.time.max()),
                }
            )
            logger.info("Completed final fit for run %s", run_name)
        logger.info("Completed training run %s", run_name)
        return TrainingResult(model, cv, final_params, tuple(working.X.columns))
