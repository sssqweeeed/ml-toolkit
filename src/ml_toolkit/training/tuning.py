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
    """Optimize model parameters using chronological validation and MLflow.

    Parameters
    ----------
    data : TrainingData
        Project-owned data with equal, nonzero row counts; finite targets;
        non-null ascending timestamps; and finite, nonnegative weights with
        positive total.
    base_params : dict[str, Any]
        Parameters merged with trial-specific overrides.
    model_factory : Callable
        Factory receiving a parameter dictionary and returning a model
        implementing the Model protocol.
    metric : Callable
        Callable receiving target, predictions, and weights as NumPy arrays;
        return a dataclass or mapping of names to numeric scalar values.
    param_space : Callable[[optuna.Trial], dict[str, Any]]
        Return parameter overrides for a trial using Optuna suggestion methods.
    metric_to_optimize : str
        Metric name present in cross-validation mean_metrics.
    n_trials : int, default 50
        Number of optimization trials.
    n_splits : int, default 5
        Number of chronological folds.
    skip_splits : int, default 0
        Number of initial folds omitted from evaluation.
    direction : {'maximize', 'minimize'}, default 'maximize'
        Optimization direction.

    Returns
    -------
    TuningResult
        Best Optuna-suggested parameters and the associated cross-validation
        result. Fixed base parameters are not included in best_params.

    Raises
    ------
    ValueError
        The requested metric is unavailable or validation settings are invalid.

    Notes
    -----
    Trial overrides take precedence over base_params. Each trial creates a
    nested MLflow run with fold runs beneath it. Configure MLflow and start a
    parent run before using this function independently.
    """
    trial_results: dict[int, CrossValidationResult] = {}
    study = optuna.create_study(direction=direction)
    logger.info(
        "Starting Optuna tuning: %s trials, metric %s, direction %s",
        n_trials,
        metric_to_optimize,
        direction,
    )

    def objective(trial: optuna.Trial) -> float:
        params = base_params | param_space(trial)
        logger.debug("Starting tuning trial %s", trial.number + 1)
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
            logger.debug(
                "Completed tuning trial %s: %s=%s",
                trial.number + 1,
                metric_to_optimize,
                value,
            )
            return value

    study.optimize(objective, n_trials=n_trials)
    logger.info(
        "Completed Optuna tuning: best %s=%s at trial %s",
        metric_to_optimize,
        study.best_value,
        study.best_trial.number + 1,
    )
    return TuningResult(study.best_params, trial_results[study.best_trial.number])
