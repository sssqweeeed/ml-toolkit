"""Predict and score train, validation and test datasets."""

import logging
from dataclasses import dataclass

import numpy as np

from ..config import ReportingConfig
from ..metrics import Metric
from ..models.base import Model
from ..tracking import runs as tracking
from ..types import FloatArray
from .contracts import DatasetSplit, PostFitEvaluator, TrainingData, TrainingStage
from .reporting import report_custom_evaluation, report_diagnostics

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationResult:
    """Store scores from the train and optional checked dataset.

    Attributes
    ----------
    train_metrics : dict[str, float]
        Training-row scores.
    heldout_metrics : dict[str, float] or None
        Validation or test scores, absent when no checked dataset is supplied.
    """

    train_metrics: dict[str, float]
    heldout_metrics: dict[str, float] | None


def calculate_metrics(
    metric: Metric, data: TrainingData, prediction: FloatArray
) -> dict[str, float]:
    """Score one dataset using the declared metric contract.

    Parameters
    ----------
    metric : Metric
        Callable returning a result with to_dict().
    data : TrainingData
        Targets and weights aligned with predictions.
    prediction : numpy.ndarray
        One finite prediction per row.

    Returns
    -------
    dict[str, float]
        Named scores, including undefined non-objective scores when applicable.
    """
    if prediction.shape != (len(data),) or not np.isfinite(prediction).all():
        raise ValueError("Predictions must contain one finite value per row")
    return metric(
        data.y.to_numpy().astype(float),
        prediction,
        data.weight.to_numpy().astype(float),
    ).to_dict()


def evaluate_model(
    model: Model,
    train: TrainingData,
    heldout: TrainingData | None,
    metric: Metric,
    reporting: ReportingConfig,
    evaluator: PostFitEvaluator | None,
    *,
    stage: TrainingStage,
    name: str,
    split: DatasetSplit = DatasetSplit.VAL,
    diagnostics: bool = False,
) -> EvaluationResult:
    """Predict once per dataset, score it and generate enabled reports.

    Parameters
    ----------
    model : Model
        Fitted adapter.
    train, heldout : TrainingData or None
        Reference training rows and optional validation/test rows.
    metric : Metric
        Scalar metric callback.
    reporting : ReportingConfig
        Built-in and custom report policy.
    evaluator : PostFitEvaluator or None
        Optional business evaluation.
    stage : {'tuning', 'validation', 'final'}
        Current workflow stage.
    name : str
        Run name used in log messages and error context.
    split : DatasetSplit, default DatasetSplit.VAL
        Identity of heldout rows.
    diagnostics : bool, default False
        Whether to generate configured built-in reports.

    Returns
    -------
    EvaluationResult
        Train scores and optional checked scores; predictions are reused for reports.
    """
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
        policy = reporting.post_fit
        if (
            evaluator is not None
            and stage in policy.stages
            and dataset_split.value in policy.splits
        ):
            report_custom_evaluation(
                evaluator=evaluator,
                data=data,
                prediction=prediction,
                split=dataset_split,
                name=name,
                write_artifacts=policy.write_artifacts,
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
