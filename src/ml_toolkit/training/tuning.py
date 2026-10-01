"""Optuna search using the same validation path as ordinary training."""

import logging
from collections.abc import Callable
from typing import Any

import mlflow
import optuna

from .contracts import TrainingData
from .cross_validation import cross_validate
from .results import CrossValidationResult, TuningResult

logger = logging.getLogger(__name__)


def tune_optuna(
    *, data: TrainingData, base_params: dict[str, Any], model_factory: Callable,
    metric: Callable, param_space: Callable[[optuna.Trial], dict[str, Any]],
    metric_to_optimize: str, n_trials: int = 50, n_splits: int = 5,
    skip_splits: int = 0, direction: str = "maximize",
) -> TuningResult:
    trial_results: dict[int, CrossValidationResult] = {}
    study = optuna.create_study(direction=direction)

    def objective(trial: optuna.Trial) -> float:
        params = base_params | param_space(trial)
        with mlflow.start_run(run_name=f"trial_{trial.number + 1}", nested=True):
            mlflow.log_params(params)
            result = cross_validate(
                data=data, params=params, model_factory=model_factory,
                metric=metric, n_splits=n_splits, skip_splits=skip_splits,
            )
            trial_results[trial.number] = result
            if metric_to_optimize not in result.mean_metrics:
                raise ValueError(
                    f"Metric {metric_to_optimize!r} is unavailable. "
                    f"Available metrics: {sorted(result.mean_metrics)}"
                )
            value = result.mean_metrics[metric_to_optimize]
            mlflow.log_metric("objective", value)
            return value

    study.optimize(objective, n_trials=n_trials)
    logger.info("Optuna best value: %s; params: %s", study.best_value, study.best_params)
    return TuningResult(study.best_params, trial_results[study.best_trial.number])
