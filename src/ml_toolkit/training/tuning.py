"""Optuna search through the shared chronological validation path."""

import logging
from collections.abc import Callable
from typing import Any

import mlflow
import optuna

from ..config import (
    CrossValidationConfig,
    PostFitEvaluationConfig,
    TrackingConfig,
    TuningConfig,
)
from ..models.base import Model
from .contracts import (
    PostFitEvaluator,
    TrainingData,
    _validate_post_fit_evaluator,
    validate_training_data,
)
from .cross_validation import cross_validate
from .results import CrossValidationResult, TuningResult

logger = logging.getLogger(__name__)


def tune_optuna(
    *,
    data: TrainingData,
    base_params: dict[str, Any],
    model_factory: Callable[[dict[str, Any]], Model],
    metric: Callable,
    param_space: Callable[[optuna.Trial], dict[str, Any]],
    config: TuningConfig,
    cv: CrossValidationConfig,
    tracking: TrackingConfig | None = None,
    post_fit_evaluator: PostFitEvaluator | None = None,
    post_fit: PostFitEvaluationConfig | None = None,
) -> TuningResult:
    """Optimize model parameters with a seeded TPE sampler and chronological CV.

    Parameters
    ----------
    data : TrainingData
        Valid chronological data, already restricted to selected features.
    base_params : dict[str, Any]
        Fixed native settings merged with each trial's overrides.
    model_factory : Callable[[dict[str, Any]], Model]
        Factory returning a fresh model adapter.
    metric : Callable
        Target/prediction/weight callable returning named scalar scores.
    param_space : Callable[[optuna.Trial], dict[str, Any]]
        Function returning trial overrides using Optuna suggestion methods.
    config : TuningConfig
        Objective name, direction, trial/time budget, and sampler seed.
    cv : CrossValidationConfig
        Shared splits reused unchanged by every trial.
    tracking : TrackingConfig or None, default None
        Trial model packaging policy; None uses defaults.
    post_fit_evaluator : PostFitEvaluator or None, default None
        Custom evaluator, invoked only when tuning is an enabled stage.
    post_fit : PostFitEvaluationConfig or None, default None
        Stage/split/image policy; None uses defaults, excluding tuning.

    Returns
    -------
    TuningResult
        Actual parameter overrides and CV result from the winning trial.
        Includes fixed values returned by param_space, not only suggestions.

    Notes
    -----
    Each trial creates nested MLflow fold runs. Configure MLflow and start a
    parent run before standalone use. No diagnostic plots run during search.
    """
    validate_training_data(data)
    post_fit = PostFitEvaluationConfig() if post_fit is None else post_fit
    evaluator = (
        post_fit_evaluator if "tuning" in post_fit.stages and post_fit.splits else None
    )
    _validate_post_fit_evaluator(data, evaluator)
    trial_results: dict[int, CrossValidationResult] = {}
    trial_overrides: dict[int, dict[str, Any]] = {}
    study = optuna.create_study(
        direction=config.direction, sampler=optuna.samplers.TPESampler(seed=config.seed)
    )
    logger.info(
        "Starting Optuna tuning: %s trials, metric %s, direction %s",
        config.n_trials,
        config.metric,
        config.direction,
    )

    def objective(trial: optuna.Trial) -> float:
        overrides = param_space(trial)
        params = base_params | overrides
        logger.debug("Starting tuning trial %s", trial.number + 1)
        with mlflow.start_run(run_name=f"trial_{trial.number + 1}", nested=True):
            mlflow.log_params(params)
            result = cross_validate(
                data=data,
                params=params,
                model_factory=model_factory,
                metric=metric,
                config=cv,
                tracking=tracking,
                post_fit_evaluator=evaluator,
                post_fit=post_fit,
                stage="tuning",
            )
            trial_results[trial.number] = result
            trial_overrides[trial.number] = overrides.copy()
            if config.metric not in result.mean_metrics:
                raise ValueError(
                    f"Metric {config.metric!r} is unavailable. Available metrics: {sorted(result.mean_metrics)}"
                )
            value = result.mean_metrics[config.metric]
            mlflow.log_metric("objective", value)
            logger.debug(
                "Completed tuning trial %s: %s=%s",
                trial.number + 1,
                config.metric,
                value,
            )
            return value

    study.optimize(objective, n_trials=config.n_trials, timeout=config.timeout_seconds)
    logger.info(
        "Completed Optuna tuning: best %s=%s at trial %s",
        config.metric,
        study.best_value,
        study.best_trial.number + 1,
    )
    return TuningResult(
        trial_overrides[study.best_trial.number], trial_results[study.best_trial.number]
    )
