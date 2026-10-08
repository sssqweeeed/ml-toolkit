from datetime import date
from pathlib import Path
from typing import assert_type

import polars as pl

from ml_toolkit.config import HoldoutConfig, TimeSeriesCVConfig, TrainingConfig
from ml_toolkit.data import Pool
from ml_toolkit.metrics import (
    Metric,
    MetricScores,
    MetricsSeverity,
    PUMetric,
    metrics_classification,
    metrics_frequency,
    metrics_severity,
)
from ml_toolkit.models import CatBoostPUClassificationModel, CatBoostRegressionModel
from ml_toolkit.models.base import Model
from ml_toolkit.training import DatasetSplit, PostFitEvaluator, TrainingData
from ml_toolkit.training.splits import TrainValSplit, make_validation_splits
from ml_toolkit.types import FloatArray


class BusinessEvaluator(PostFitEvaluator):
    def evaluate(
        self,
        *,
        object_id: pl.Series,
        y_pred: FloatArray,
        y_true: FloatArray,
        split: DatasetSplit,
    ) -> dict[str, float]:
        return {"rows": float(len(object_id))}

    def write_artifacts(
        self,
        *,
        object_id: pl.Series,
        y_pred: FloatArray,
        y_true: FloatArray,
        split: DatasetSplit,
        artifact_dir: Path,
    ) -> list[Path]:
        return []


def dictionary_metric(
    y: FloatArray, predictions: FloatArray, weight: FloatArray
) -> MetricScores:
    return MetricScores({"mae": float(abs(y - predictions).mean())})


def severity_metric(
    y: FloatArray, predictions: FloatArray, weight: FloatArray
) -> MetricsSeverity:
    return metrics_severity(y, predictions, weight, power=1.5)


def contracts(data: Pool) -> None:
    protocol: TrainingData = data
    model: Model = CatBoostRegressionModel({"iterations": 10})
    pu = CatBoostPUClassificationModel(
        {"loss": "uPU", "pu_prior": 0.1, "task_type": "GPU", "devices": "0"}
    )
    assert_type(pu.fit(data), CatBoostPUClassificationModel)
    assert_type(pu.predict_pool(data), FloatArray)
    metrics: tuple[Metric, ...] = (
        dictionary_metric,
        metrics_classification,
        metrics_frequency,
        severity_metric,
        PUMetric(0.2),
    )
    assert_type(
        make_validation_splits(data, HoldoutConfig()), list[TrainValSplit[Pool]]
    )
    assert_type(
        make_validation_splits(data, TimeSeriesCVConfig(n_splits=3)),
        list[TrainValSplit[Pool]],
    )
    assert_type(protocol.object_id, pl.Series | None)
    assert_type(model.get_refit_params(), dict[str, object])
    assert_type(
        metrics[0](
            data.y.to_numpy(), data.y.to_numpy(), data.weight.to_numpy()
        ).to_dict(),
        dict[str, float],
    )
    TrainingConfig(validation=HoldoutConfig())
    TrainingConfig(validation=HoldoutConfig(), test_size=4)
    TrainingConfig(validation=HoldoutConfig(), test_size=0.2)
    TrainingConfig(validation=HoldoutConfig(), test_size="2025-01-01")
    TrainingConfig(validation=HoldoutConfig(), test_size=date(2025, 1, 1))
    TrainingConfig(validation=HoldoutConfig(), test_size=None)


from ml_toolkit.config import RFEFeatureSelectionConfig
from ml_toolkit.training import (
    FeatureSelectionResult,
    importance_ranker,
    select_features,
)


def explicit_selection(data: Pool) -> None:
    settings = RFEFeatureSelectionConfig(
        min_features=1, metric="mae", direction="minimize"
    )
    assert_type(
        select_features(
            train=data,
            validation=data,
            params={},
            model_factory=CatBoostRegressionModel,
            metric=dictionary_metric,
            ranker=importance_ranker,
            config=settings,
        ),
        FeatureSelectionResult,
    )


from ml_toolkit.config import CommonPlotsConfig, FeatureLevelSpec, PUPlotsConfig
from ml_toolkit.feature_levels import LevelTransformer
from ml_toolkit.plots.regression import observed_fitted_by_level
from ml_toolkit.psi import PSIResult, compare_psi
import pandas as pd


def feature_diagnostics(
    reference: pl.DataFrame, current: pl.DataFrame, y: FloatArray, output: Path
) -> None:
    common = CommonPlotsConfig(
        features=(FeatureLevelSpec("age", "numerical", lower_bound=0),),
        psi=False,
        observed_fitted=True,
    )
    assert_type(common.features, tuple[FeatureLevelSpec, ...] | None)
    assert_type(PUPlotsConfig(common=common), PUPlotsConfig)
    transformer = LevelTransformer((FeatureLevelSpec("age", "numerical"),)).fit(
        reference
    )
    reference_levels = transformer.transform(reference)
    current_levels = transformer.transform(current)
    assert transformer.config is not None
    assert_type(
        compare_psi(reference_levels, current_levels, transformer.config), PSIResult
    )
    assert_type(
        observed_fitted_by_level(
            reference_levels["age"],
            y,
            y,
            current_levels["age"],
            y,
            y,
            output,
            level_order=transformer.config.features[0].levels,
        ),
        pd.DataFrame,
    )
