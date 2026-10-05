"""One reusable fitting workflow called by project-specific training recipes."""

import logging
from collections.abc import Callable
from typing import Any

import mlflow
import optuna

from .contracts import (
    DatasetSplit, PostFitEvaluator, TrainingData, _validate_post_fit_evaluator,
    validate_training_data,
)
from .cross_validation import _time_series_splits, cross_validate
from .evaluation import DiagnosticsOptions, _log_post_fit_evaluation
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
    post_fit_evaluator: PostFitEvaluator | None = None,
) -> TrainingResult:
    """Select features, tune, validate, and fit a model with MLflow logging.

    Parameters
    ----------
    data : TrainingData
        Project-owned data with equal, nonzero row counts; finite targets;
        non-null ascending timestamps; and finite, nonnegative weights with
        positive total.
    params : dict[str, Any]
        Initial model parameters. The input dictionary is copied.
    model_factory : Callable
        Factory receiving a parameter dictionary and returning a model
        implementing the Model protocol.
    metric : Callable
        Callable receiving target, predictions, and weights as NumPy arrays;
        return a dataclass or mapping of names to numeric scalar values.
    run_name : str
        Name of the MLflow run created for this workflow.
    n_splits : int
        Number of expanding-window time-series folds; at least two.
    skip_splits : int, default 0
        Number of initial folds to omit; must be smaller than n_splits.
    diagnostics : DiagnosticsOptions or None, default None
        Optional PSI tables and diagnostic plots for evaluated folds.
    select_features : int or None, default None
        Number of features to retain. Selection uses only training rows of the
        first evaluated fold, then reuses the subset throughout the workflow.
    tune_params : bool, default False
        Run Optuna search before cross-validation and final fitting.
    param_space : Callable or None, default None
        Function accepting an Optuna Trial and returning model parameter
        overrides; required when tuning.
    metric_to_optimize : str or None, default None
        Name returned by metric to optimize; required when tuning.
    optimize_direction : {'maximize', 'minimize'}, default 'maximize'
        Direction of Optuna optimization.
    n_trials : int, default 50
        Number of Optuna trials when tuning is enabled.
    post_fit_evaluator : PostFitEvaluator or None, default None
        User-created evaluator receiving aligned IDs, model outputs, targets,
        and DatasetSplit. Requires non-null string or integer data.object_id.
        The same instance evaluates train/val for CV and tuning folds, including
        repeated CV, and train after final fitting. Results are logged under
        <split>/custom/ and excluded from ordinary CV metrics and optimization.
        The toolkit does not copy or reset evaluator state. Feature-selection
        fits do not invoke it; test data are not evaluated by this workflow.

    Returns
    -------
    TrainingResult
        Fitted model, fold results, final parameters, and retained feature
        names.

    Raises
    ------
    ValueError
        Training data or split settings are invalid, selection has too few
        rows, required tuning arguments are missing, or post-fit IDs, prediction
        shapes, metric names, or finite metric values are invalid.
    TypeError
        The evaluator has no evaluate method, IDs have an unsupported type, or
        its result is not a dictionary of floating scalars.

    Notes
    -----
    The caller owns data preparation, model settings, and MLflow configuration.
    This function creates a run, nested when a run is already active, logs each
    stage, and fits the final model on all retained rows. When folds report
    best iterations, their median zero-based index plus one replaces an
    existing iterations, n_estimators, or num_trees setting for final fitting.
    Diagnostics after tuning evaluate the winning parameters again.

    PU models require a matching PUMetric and, if enabled, diagnostics with
    task="pu_classification".

    Examples
    --------
    Prepare a project-owned data object and configure MLflow before calling:

    >>> from ml_toolkit.models import CatBoostRegressionModel
    >>> from ml_toolkit.metrics import metrics_frequency
    >>> from ml_toolkit.training import train_model
    >>> result = train_model(  # doctest: +SKIP
    ...     data=data,
    ...     params={"iterations": 100, "loss_function": "Poisson"},
    ...     model_factory=CatBoostRegressionModel,
    ...     metric=metrics_frequency,
    ...     run_name="frequency",
    ...     n_splits=3,
    ... )
    >>> result.cross_validation.mean_metrics  # doctest: +SKIP
    """
    validate_training_data(data)
    _validate_post_fit_evaluator(data, post_fit_evaluator)
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
        _validate_post_fit_evaluator(working, post_fit_evaluator)
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
                post_fit_evaluator=post_fit_evaluator,
            )
            final_params.update(tuning.best_params)
            cv = tuning.cross_validation
            # The selected parameters need diagnostic plots as well as metrics.
            if diagnostics is not None:
                cv = cross_validate(
                    data=working, params=final_params, model_factory=model_factory,
                    metric=metric, n_splits=n_splits, skip_splits=skip_splits,
                    diagnostics=diagnostics,
                    post_fit_evaluator=post_fit_evaluator,
                )
        else:
            cv = cross_validate(
                data=working, params=final_params, model_factory=model_factory,
                metric=metric, n_splits=n_splits, skip_splits=skip_splits,
                diagnostics=diagnostics,
                post_fit_evaluator=post_fit_evaluator,
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
            if post_fit_evaluator is not None:
                _log_post_fit_evaluation(
                    evaluator=post_fit_evaluator, data=working,
                    prediction=model.predict_pool(working),
                    split=DatasetSplit.TRAIN, name="final_fit",
                )
            model.log(working)
            mlflow.log_params({
                "train_n_rows": len(working),
                "train_date_start": str(working.time.min()),
                "train_date_end": str(working.time.max()),
            })
            logger.info("Completed final fit for run %s", run_name)
        logger.info("Completed training run %s", run_name)
        return TrainingResult(model, cv, final_params, tuple(working.X.columns))
