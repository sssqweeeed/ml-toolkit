from pathlib import Path
from typing import Callable

import polars as pl

from ml_toolkit.config import (
    PUPlotsConfig,
    RegressionDiagnosticsConfig,
    TrainingConfig,
    TuningConfig,
)
from ml_toolkit.metrics import Metric, MetricScores
from ml_toolkit.training import DatasetSplit, PostFitEvaluator
from ml_toolkit.types import FloatArray, ModelParams

TrainingConfig(validation={"n_splits": 3})  # expected-type-error
TuningConfig(metric="mae", direction="invalid")  # expected-type-error
RegressionDiagnosticsConfig(plots=PUPlotsConfig())  # expected-type-error
MetricScores({"mae": "1"})  # expected-type-error


def dictionary_metric(
    y: FloatArray, predictions: FloatArray, weight: FloatArray
) -> dict[str, float]:
    return {"mae": 1.0}


metric: Metric = dictionary_metric  # expected-type-error


class WrongWriter(PostFitEvaluator):
    def evaluate(
        self,
        *,
        object_id: pl.Series,
        y_pred: FloatArray,
        y_true: FloatArray,
        split: DatasetSplit,
    ) -> dict[str, float]:
        return {"score": "1"}  # expected-type-error

    def write_artifacts(
        self,
        *,
        object_id: pl.Series,
        y_pred: FloatArray,
        y_true: FloatArray,
        split: DatasetSplit,
        artifact_dir: Path,
    ) -> list[Path]:
        return ["figure.png"]  # expected-type-error


score: Callable[[ModelParams], float] = lambda params: {"mae": 1.0}  # expected-type-error
split: DatasetSplit = "val"  # expected-type-error

from ml_toolkit.config import TimeSeriesCVConfig
from ml_toolkit.models.base import Model
from ml_toolkit.training import FeatureRanker

factory: Callable[[ModelParams], Model] = lambda params: {"model": 1.0}  # expected-type-error
ranker: FeatureRanker = lambda model, data: [1.0]  # expected-type-error
TimeSeriesCVConfig(3)  # expected-type-error

from ml_toolkit.config import HoldoutConfig, RFEFeatureSelectionConfig

RFEFeatureSelectionConfig(min_features=1, metric="mae", direction="minimize", holdout=HoldoutConfig())  # expected-type-error
