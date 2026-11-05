"""Predict and score train, validation and test datasets."""

from dataclasses import dataclass

import numpy as np

from .._console import Logger, metrics
from ..psi import PSIBaseline
from ..config import ReportingConfig
from ..evaluation import PredictionEvaluator
from ..data_analysis import DataAnalyzer
from ..evaluation._inputs import combine_evaluation_pools as _combine_pools
import polars as pl
from ..metrics import Metric
from ..metrics._evaluation import calculate_metrics as calculate_metrics
from ..model_evaluation import ModelEvaluator
from ..models.base import Model
from ..tracking import runs as tracking
from ..types import FloatArray
from .contracts import DatasetSplit, Pool, TrainingStage
from .reporting import evaluate_report_metrics, report_predictions

logger = Logger(__name__)


@dataclass(frozen=True)
class EvaluationResult:
    train_metrics: dict[str, float]
    heldout_metrics: dict[str, float] | None
    objective_metrics: dict[str, float] | None = None


def evaluate_model(
    model: Model,
    train: Pool,
    heldout: Pool | None,
    metric: Metric,
    reporting: ReportingConfig,
    evaluators: tuple[PredictionEvaluator, ...],
    *,
    analyzers: tuple[DataAnalyzer, ...] = (),
    model_evaluators: tuple[ModelEvaluator, ...] = (),
    stage: TrainingStage,
    name: str,
    split: DatasetSplit = DatasetSplit.VAL,
    psi_baseline: PSIBaseline | None = None,
    objective_evaluator: PredictionEvaluator | None = None,
) -> EvaluationResult:
    datasets = [(DatasetSplit.TRAIN, train)]
    if heldout is not None:
        datasets.append((split, heldout))
    scores: dict[DatasetSplit, dict[str, float]] = {}
    predictions: dict[DatasetSplit, FloatArray] = {}
    for dataset_split, data in datasets:
        prediction = model.predict_pool(data)
        predictions[dataset_split] = prediction
        scores[dataset_split] = calculate_metrics(metric, data, prediction)
        tracking.log_params(
            {
                f"{dataset_split.value}_n_rows": len(data),
                f"{dataset_split.value}_date_start": str(data.time.min()),
                f"{dataset_split.value}_date_end": str(data.time.max()),
            }
        )
        tracking.log_metrics(
            {
                f"{dataset_split.value}/{key}": value
                for key, value in scores[dataset_split].items()
            }
        )
        logger.debug(
            "Evaluation %s %s metrics: %s",
            name,
            dataset_split.value,
            scores[dataset_split],
            console=stage == "tuning",
        )
    if stage != "tuning":
        metrics(
            f"{name} metrics",
            {part.value: values for part, values in scores.items()},
        )
    enabled = {
        "tuning": reporting.on_tuning,
        "validation": reporting.on_validation,
        "test": reporting.on_test,
        "final_fit": reporting.on_final_fit,
    }[stage]
    reports_enabled = enabled and bool(
        evaluators or model_evaluators or analyzers or reporting.psi or reporting.shap is not None
    )
    objective_metrics = None
    if reports_enabled or objective_evaluator is not None:
        data = _combine_pools(train, heldout)
        prediction = (
            predictions[DatasetSplit.TRAIN]
            if heldout is None
            else np.concatenate([predictions[part] for part, _ in datasets])
        )
        labels = pl.concat(
            [
                pl.repeat(part.value, len(pool), dtype=pl.String, eager=True)
                for part, pool in datasets
            ]
        ).rename("split")
        if objective_evaluator is not None:
            objective_metrics = evaluate_report_metrics(
                name=objective_evaluator.name,
                evaluate=lambda: objective_evaluator.evaluate(
                    data=data, prediction=prediction, split=labels
                ),
                context=f"Prediction evaluation evaluator={objective_evaluator.name}, stage={name}",
            )
        if reports_enabled:
            report_predictions(
                model=model,
                data=data,
                prediction=prediction,
                split=labels,
                evaluators=evaluators,
                model_evaluators=model_evaluators,
                metric=metric,
                analyzers=analyzers,
                config=reporting,
                stage=name,
                psi_baseline=psi_baseline,
                scored_evaluator=objective_evaluator,
            )
    return EvaluationResult(scores[DatasetSplit.TRAIN], scores.get(split), objective_metrics)
