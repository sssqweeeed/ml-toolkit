"""Predict and score train, validation and test datasets."""

import logging
from dataclasses import dataclass

import numpy as np

from ..psi import PSIBaseline
from ..config import ReportingConfig
from ..evaluation import PredictionEvaluator
from ..data_analysis import DataAnalyzer
import polars as pl
from ..metrics import Metric
from ..models.base import Model
from ..tracking import runs as tracking
from ..types import FloatArray
from .contracts import DatasetSplit, Pool, TrainingStage
from .reporting import report_predictions

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationResult:
    train_metrics: dict[str, float]
    heldout_metrics: dict[str, float] | None


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
    evaluators: tuple[PredictionEvaluator, ...],
    *,
    analyzers: tuple[DataAnalyzer, ...] = (),
    stage: TrainingStage,
    name: str,
    split: DatasetSplit = DatasetSplit.VAL,
    psi_baseline: PSIBaseline | None = None,
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
    enabled = {
        "tuning": reporting.on_tuning,
        "validation": reporting.on_validation,
        "test": reporting.on_test,
        "final_fit": reporting.on_final_fit,
    }[stage]
    if enabled and (
        evaluators or analyzers or reporting.psi or reporting.shap is not None
    ):
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
        report_predictions(
            model=model,
            data=data,
            prediction=prediction,
            split=labels,
            evaluators=evaluators,
            analyzers=analyzers,
            config=reporting,
            stage=name,
            psi_baseline=psi_baseline,
        )
    return EvaluationResult(scores[DatasetSplit.TRAIN], scores.get(split))


def _combine_pools(train: Pool, heldout: Pool | None) -> Pool:
    """Join standard pool fields positionally, preserving feature metadata."""
    if heldout is None:
        return train
    if (
        train.X.schema != heldout.X.schema
        or train.cat_features != heldout.cat_features
        or train.numerical_features != heldout.numerical_features
    ):
        raise ValueError("Reporting pools must have matching features and metadata")

    def optional(left: pl.Series | None, right: pl.Series | None) -> pl.Series | None:
        if left is None and right is None:
            return None
        if left is None or right is None:
            raise ValueError("Reporting pools must have matching optional fields")
        return pl.concat([left, right], rechunk=False)

    return Pool(
        X=pl.concat([train.X, heldout.X], rechunk=False),
        y=pl.concat([train.y, heldout.y], rechunk=False),
        time=pl.concat([train.time, heldout.time], rechunk=False),
        weight=pl.concat([train.weight, heldout.weight], rechunk=False),
        business_type=optional(train.business_type, heldout.business_type),
        object_id=optional(train.object_id, heldout.object_id),
        cat_features=train.cat_features,
        numerical_features=train.numerical_features,
    )
