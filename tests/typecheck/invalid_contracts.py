from ml_toolkit.feature_selection import RFESelector, TopKSelector, ForwardSelector, FeatureSelector
from ml_toolkit.types import TimePeriod
from ml_toolkit.training import importance_ranker
from pathlib import Path
from typing import Callable

import polars as pl

from ml_toolkit import Pool, TrainingResult, train_model
from ml_toolkit.models import CatBoostRegressionModel
from ml_toolkit.config import (
    PUPlotsConfig,
    RegressionDiagnosticsConfig,
    ReportingConfig,
    TrainingConfig,
    TuningConfig,
)
from ml_toolkit.metrics import Metric, MetricScores
from ml_toolkit.training import DatasetSplit, PostFitEvaluator
from ml_toolkit.types import FloatArray, ModelParams

TrainingConfig(validation={"n_splits": 3})  # expected-type-error
TuningConfig(metric="mae", direction="invalid")  # expected-type-error
RegressionDiagnosticsConfig(plots=PUPlotsConfig())  # expected-type-error
ReportingConfig(post_fit=False)  # expected-type-error
ReportingConfig(post_fit={"stages": ()})  # expected-type-error
MetricScores({"mae": "1"})  # expected-type-error
result: TrainingResult = {}  # expected-type-error


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


WrongWriter()  # expected-type-error
WrongWriter(name=1)  # expected-type-error
wrong_evaluator = WrongWriter(name="business")
wrong_evaluator.name = "quality"  # expected-type-error


def invalid_evaluator_list(data: Pool, metric: Metric) -> None:
    train_model(
        data=data,
        params={},
        model_factory=CatBoostRegressionModel,
        metric=metric,
        run_name="invalid-evaluators",
        config=TrainingConfig(validation=TimeSeriesCVConfig(n_splits=2)),
        post_fit_evaluator=["business"],  # expected-type-error
    )


score: Callable[[ModelParams], float] = lambda params: {"mae": 1.0}  # expected-type-error
split: DatasetSplit = "val"  # expected-type-error

from ml_toolkit.config import TimeSeriesCVConfig
from ml_toolkit.models.base import Model
from ml_toolkit.training import FeatureRanker

factory: Callable[[ModelParams], Model] = lambda params: {"model": 1.0}  # expected-type-error
ranker: FeatureRanker = lambda model, data: [1.0]  # expected-type-error
TimeSeriesCVConfig(3)  # expected-type-error

from ml_toolkit.config import HoldoutConfig

TrainingConfig(validation=HoldoutConfig(), test_size=["2025-01-01"])  # expected-type-error
TrainingConfig(validation=HoldoutConfig(), test_size={"start_date": "2025-01-01"})  # expected-type-error
RFESelector(ranker=importance_ranker, min_features=1, metric="mae", direction="minimize", holdout=HoldoutConfig())  # expected-type-error

from ml_toolkit.config import CommonPlotsConfig, FeatureLevelSpec

CommonPlotsConfig(features=("age",))  # expected-type-error
CommonPlotsConfig(psi="on")  # expected-type-error
CommonPlotsConfig(observed_fitted=None)  # expected-type-error
FeatureLevelSpec("age", "continuous")  # expected-type-error
RegressionDiagnosticsConfig(psi=None)  # expected-type-error

Pool()  # expected-type-error


class IncompleteProjectPool:
    X: pl.DataFrame
    y: pl.Series
    time: pl.Series
    weight: pl.Series

    def __len__(self) -> int:
        return len(self.X)


incomplete_pool: Pool = IncompleteProjectPool()  # expected-type-error


TopKSelector(k=1)  # expected-type-error
ForwardSelector(metric="mae", direction="lowest", max_features=1, max_fits=2)  # expected-type-error
period: TimePeriod = "fortnight"  # expected-type-error
CommonPlotsConfig(period="seconds")  # expected-type-error
TrainingConfig(feature_selection=None)  # expected-type-error


class WrongSelector:
    def select(self, *, train: Pool) -> tuple[str, ...]:
        return tuple(train.X.columns)


selector: FeatureSelector = WrongSelector()  # expected-type-error
