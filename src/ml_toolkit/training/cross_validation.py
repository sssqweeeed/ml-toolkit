"""Time-series validation with MLflow logging per fold."""

import logging
from collections.abc import Callable
from typing import Any

import mlflow
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from .contracts import TrainingData, validate_training_data
from .evaluation import DiagnosticsOptions, log_evaluation
from .results import CrossValidationResult

logger = logging.getLogger(__name__)


def cross_validate(
    *, data: TrainingData, params: dict[str, Any], model_factory: Callable,
    metric: Callable, n_splits: int, skip_splits: int = 0,
    diagnostics: DiagnosticsOptions | None = None,
) -> CrossValidationResult:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if not 0 <= skip_splits < n_splits:
        raise ValueError("skip_splits must be between 0 and n_splits - 1")
    validate_training_data(data)
    fold_metrics: list[dict[str, float]] = []
    best_iterations: list[int] = []
    splitter = TimeSeriesSplit(n_splits=n_splits)
    for fold_index, (train_index, validation_index) in enumerate(
        splitter.split(np.arange(len(data))), start=1,
    ):
        if fold_index <= skip_splits:
            logger.info("Skipping fold %s", fold_index)
            continue
        train = data.take(train_index)
        validation = data.take(validation_index)
        with mlflow.start_run(run_name=f"fold_{fold_index}", nested=True):
            mlflow.log_params(params)
            model = model_factory(params)
            model.fit(train, validation)
            scores = log_evaluation(
                model=model, train=train, validation=validation,
                train_prediction=model.predict_pool(train),
                validation_prediction=model.predict_pool(validation),
                metric=metric, diagnostics=diagnostics, name=f"fold_{fold_index}",
            )
            model.log(validation)
            if model.best_iteration is not None:
                best_iterations.append(int(model.best_iteration))
            fold_metrics.append(scores["val"])
    if not fold_metrics:
        raise ValueError("No folds were processed")
    result = CrossValidationResult(tuple(fold_metrics), tuple(best_iterations))
    mlflow.log_metrics({f"cv/{key}": value for key, value in result.mean_metrics.items()})
    return result
