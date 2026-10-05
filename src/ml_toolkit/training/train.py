"""One reusable fitting workflow called by project-specific training recipes."""

import logging
from collections.abc import Callable
from typing import Any

import mlflow
import optuna

from .contracts import TrainingData, validate_training_data
from .cross_validation import _time_series_splits, cross_validate
from .evaluation import DiagnosticsOptions
from .feature_selection import feature_elimination
from .results import TrainingResult
from .tuning import tune_optuna

logger = logging.getLogger(__name__)


def train_model(
    *, data: TrainingData, params: dict[str, Any], model_factory: Callable,
    metric: Callable, run_name: str, n_splits: int, skip_splits: int = 0,
    diagnostics: DiagnosticsOptions | None = None,
    select_features: int | None = None, tune_params: bool = False,
    param_space: Callable[[optuna.Trial], dict[str, Any]] | None = None,
    metric_to_optimize: str | None = None,
    optimize_direction: str = "maximize", n_trials: int = 50,
) -> TrainingResult:
    """Select, tune, validate and fit, logging every stage to active MLflow tracking.

    The caller owns ``data`` and all settings. Model factories receive a plain
    parameter dictionary. An existing MLflow run can contain this workflow.
    """
    validate_training_data(data)
    logger.info("Starting training run %s", run_name)
    logger.debug("Training data: %s rows, %s features", len(data), len(data.X.columns))
    with mlflow.start_run(run_name=run_name, nested=mlflow.active_run() is not None):
        working = data
        if select_features is not None:
            selection_index = next(
                train_index
                for fold_index, (train_index, _) in _time_series_splits(
                    len(data), n_splits, skip_splits,
                )
                if fold_index > skip_splits
            )
            if len(selection_index) < 2:
                raise ValueError(
                    "First evaluated fold has fewer than two training rows "
                    "for feature selection"
                )
            selected = feature_elimination(
                data=data.take(selection_index), params=params, model_factory=model_factory,
                num_features_to_select=select_features,
            )
            working = data.select_features(selected.X.columns)
            mlflow.log_dict({"features": working.X.columns}, "selection/features.json")
        final_params = params.copy()
        if tune_params:
            if param_space is None or metric_to_optimize is None:
                raise ValueError("param_space and metric_to_optimize are required for tuning")
            tuning = tune_optuna(
                data=working, base_params=final_params, model_factory=model_factory,
                metric=metric, param_space=param_space,
                metric_to_optimize=metric_to_optimize, n_trials=n_trials,
                n_splits=n_splits, skip_splits=skip_splits,
                direction=optimize_direction,
            )
            final_params.update(tuning.best_params)
            cv = tuning.cross_validation
            # The selected parameters need diagnostic plots as well as metrics.
            if diagnostics is not None:
                cv = cross_validate(
                    data=working, params=final_params, model_factory=model_factory,
                    metric=metric, n_splits=n_splits, skip_splits=skip_splits,
                    diagnostics=diagnostics,
                )
        else:
            cv = cross_validate(
                data=working, params=final_params, model_factory=model_factory,
                metric=metric, n_splits=n_splits, skip_splits=skip_splits,
                diagnostics=diagnostics,
            )
        selected_iterations = cv.final_iterations
        if selected_iterations is not None:
            for name in ("iterations", "n_estimators", "num_trees"):
                if name in final_params:
                    final_params[name] = selected_iterations
                    break
        final_params.pop("early_stopping_rounds", None)
        if "use_best_model" in final_params:
            final_params["use_best_model"] = False
        with mlflow.start_run(run_name="final_fit", nested=True):
            logger.info("Starting final fit for run %s", run_name)
            logger.debug("Final fit selected iterations: %s", selected_iterations)
            mlflow.log_params(final_params)
            if selected_iterations is not None:
                mlflow.log_param("selected_iterations", selected_iterations)
            model = model_factory(final_params)
            model.fit(working)
            model.log(working)
            mlflow.log_params({
                "train_n_rows": len(working),
                "train_date_start": str(working.time.min()),
                "train_date_end": str(working.time.max()),
            })
            logger.info("Completed final fit for run %s", run_name)
        logger.info("Completed training run %s", run_name)
        return TrainingResult(model, cv, final_params, tuple(working.X.columns))
