"""Predict and score train, validation and test datasets."""

import logging
from dataclasses import dataclass

import numpy as np

from ..config import PredictionEvaluationConfig, ReportingConfig
from ..metrics import Metric
from ..models.base import Model
from ..tracking import runs as tracking
from ..types import FloatArray
from .contracts import DatasetSplit, PredictionEvaluator, Pool, TrainingStage
from .reporting import report_custom_evaluation, report_diagnostics

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationResult:
    train_metrics: dict[str, float]
    heldout_metrics: dict[str, float] | None


@dataclass(frozen=True)
class PreparedPredictionEvaluation:
    evaluators: tuple[PredictionEvaluator, ...]
    calls: frozenset[tuple[TrainingStage, DatasetSplit]]
    write_artifacts: bool


def prepare_prediction_evaluation(
    data: Pool,
    evaluator: PredictionEvaluator | list[PredictionEvaluator] | None,
    policy: PredictionEvaluationConfig | None,
    *,
    tuning: bool,
    has_test: bool,
) -> PreparedPredictionEvaluation | None:
    if evaluator is None:
        return None
    evaluators = tuple(evaluator) if isinstance(evaluator, list) else (evaluator,)
    names: set[str] = set()
    for item in evaluators:
        name = item.name.casefold()
        if name in names:
            raise ValueError(f"Evaluator names must be unique: {item.name!r}")
        names.add(name)
    if not evaluators:
        return None
    policy = PredictionEvaluationConfig() if policy is None else policy
    available: set[tuple[TrainingStage, DatasetSplit]] = {
        ("validation", DatasetSplit.TRAIN),
        ("validation", DatasetSplit.VAL),
        ("final", DatasetSplit.TRAIN),
    }
    if tuning:
        available.update(
            {("tuning", DatasetSplit.TRAIN), ("tuning", DatasetSplit.VAL)}
        )
    if has_test:
        available.add(("final", DatasetSplit.TEST))
    calls: frozenset[tuple[TrainingStage, DatasetSplit]] = frozenset(
        (stage, split)
        for stage, split in available
        if stage in policy.stages and split.value in policy.splits
    )
    if not calls:
        return None
    if data.object_id is None:
        raise ValueError("object_id is required when prediction_evaluators is supplied")
    return PreparedPredictionEvaluation(evaluators, calls, policy.write_artifacts)


def calculate_metrics(
    metric: Metric, data: Pool, prediction: FloatArray
) -> dict[str, float]:
    if prediction.shape != (len(data),) or not np.isfinite(prediction).all():
        raise ValueError("Predictions must contain one finite value per row")
    return metric(
        data.y.to_numpy().astype(float),
        prediction,
        data.weight.to_numpy().astype(float),
    ).to_dict()


def evaluate_model(
    model: Model,
    train: Pool,
    heldout: Pool | None,
    metric: Metric,
    reporting: ReportingConfig,
    prediction_evaluation: PreparedPredictionEvaluation | None,
    *,
    stage: TrainingStage,
    name: str,
    split: DatasetSplit = DatasetSplit.VAL,
    diagnostics: bool = False,
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
        )
        if (
            prediction_evaluation is not None
            and (stage, dataset_split) in prediction_evaluation.calls
        ):
            for evaluator in prediction_evaluation.evaluators:
                report_custom_evaluation(
                    evaluator=evaluator,
                    data=data,
                    prediction=prediction,
                    split=dataset_split,
                    name=name,
                    write_artifacts=prediction_evaluation.write_artifacts,
                )
    if diagnostics and reporting.diagnostics is not None and heldout is not None:
        report_diagnostics(
            model,
            train,
            heldout,
            predictions[DatasetSplit.TRAIN],
            predictions[split],
            reporting.diagnostics,
            name,
            split,
        )
    return EvaluationResult(scores[DatasetSplit.TRAIN], scores.get(split))
