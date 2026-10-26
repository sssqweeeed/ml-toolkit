from ml_toolkit.feature_selection import (
    RFESelector,
    TopKSelector,
    ForwardSelector,
    FeatureSelector,
)
from ml_toolkit.types import TimePeriod
from ml_toolkit.training import importance_ranker
from pathlib import Path
from typing import Callable

import polars as pl

from ml_toolkit import Pool, TrainingResult, train_model
from ml_toolkit.models import CatBoostRegressionModel
from ml_toolkit.config import (
    ReportingConfig,
    TrainingConfig,
    TuningConfig,
)
from ml_toolkit.metrics import Metric, MetricScores
from ml_toolkit.training import DatasetSplit
from ml_toolkit.evaluation import PredictionEvaluator
from ml_toolkit.config import FigureConfig, FeatureLevelConfig
from ml_toolkit.types import FloatArray, ModelParams

TrainingConfig(validation={'n_splits': 3})  # expected-type-error
TrainingConfig(verbosity='invalid')  # expected-type-error
TrainingConfig(verbosity=True)  # expected-type-error
TuningConfig(metric='mae', direction='invalid')  # expected-type-error
ReportingConfig(figure=False)  # expected-type-error
ReportingConfig(on_tuning='yes')  # expected-type-error
ReportingConfig(on_final_fit=None)  # expected-type-error
MetricScores({'mae': '1'})  # expected-type-error
result: TrainingResult = {}  # expected-type-error


def dictionary_metric(
    y: FloatArray, predictions: FloatArray, weight: FloatArray
) -> dict[str, float]:
    return {"mae": 1.0}


metric: Metric = dictionary_metric  # expected-type-error


class WrongWriter(PredictionEvaluator):
    def evaluate(
        self,
        *,
        data: Pool,
        prediction: FloatArray,
        split: pl.Series,
    ) -> dict[str, float]:
        return {"score": "1"}  # expected-type-error

    def write_artifacts(
        self,
        *,
        data: Pool,
        prediction: FloatArray,
        split: pl.Series,
        artifact_dir: Path,
        figure: FigureConfig,
        features: tuple[FeatureLevelConfig, ...] | None,
        n_bins: int,
    ) -> list[Path]:
        return ["figure.png"]  # expected-type-error


WrongWriter()  # expected-type-error
WrongWriter(name=1)  # expected-type-error
wrong_evaluator = WrongWriter(name="business")
wrong_evaluator.name = 'quality'  # expected-type-error


def invalid_evaluator_list(data: Pool, metric: Metric) -> None:
    train_model(splitter=TimeSeriesCV(data, n_splits=2), params={}, model_factory=CatBoostRegressionModel, metric=metric, run_name='invalid-evaluators', config=TrainingConfig(), prediction_evaluators=['business'])  # expected-type-error


score: Callable[[ModelParams], float] = lambda params: {'mae': 1.0}  # expected-type-error
split: DatasetSplit = 'val'  # expected-type-error

from ml_toolkit.training import Holdout, Split, Splitter, TimeSeriesCV
from ml_toolkit.models.base import Model
from ml_toolkit.training import FeatureRanker

factory: Callable[[ModelParams], Model] = lambda params: {'model': 1.0}  # expected-type-error
ranker: FeatureRanker = lambda model, data: [1.0]  # expected-type-error
TimeSeriesCV(3)  # expected-type-error


def invalid_test_sizes(data: Pool) -> None:
    data.to_model_input(categorical_as_strings="yes")  # expected-type-error
    Holdout(data, test_size=["2025-01-01"])  # expected-type-error
    Holdout(data, test_size={"start_date": "2025-01-01"})  # expected-type-error

RFESelector(ranker=importance_ranker, min_features=1, metric='mae', direction='minimize', holdout=None)  # expected-type-error

from ml_toolkit.config import FeatureLevelConfig

ReportingConfig(features=('age',))  # expected-type-error
ReportingConfig(psi='on')  # expected-type-error
ReportingConfig(n_bins='ten')  # expected-type-error
FeatureLevelConfig('age', 'continuous')  # expected-type-error
ReportingConfig(psi=None)  # expected-type-error

from ml_toolkit.feature_levels import FeatureLevels
from ml_toolkit.psi import compare_psi

rules = FeatureLevelConfig("age", "numerical")
FeatureLevels(config=rules, levels=(), boundaries=('1',))  # expected-type-error
compare_psi(pl.DataFrame(), pl.DataFrame(), feature_levels=(rules,))  # expected-type-error

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
ForwardSelector(metric='mae', direction='lowest', max_features=1, max_fits=2)  # expected-type-error
period: TimePeriod = 'fortnight'  # expected-type-error
ReportingConfig(period='seconds')  # expected-type-error
TrainingConfig(feature_selection=None)  # expected-type-error


class WrongSelector:
    def select(self, *, train: Pool) -> tuple[str, ...]:
        return tuple(train.X.columns)


selector: FeatureSelector = WrongSelector()  # expected-type-error


from collections.abc import Iterator


class WrongSplitter(Splitter[Pool]):
    def get_validation(self) -> Iterator[Pool]:  # expected-type-error
        yield self.data

    def get_test(self) -> Pool:  # expected-type-error
        return self.data


def invalid_splitter(data: Pool) -> None:
    Split(train=data, heldout='test')  # expected-type-error
    Splitter(data)  # expected-type-error
    train_model(splitter=data, params={}, model_factory=CatBoostRegressionModel, metric=metrics, run_name='invalid', config=TrainingConfig())  # expected-type-error


metrics: Metric = lambda y, prediction, weight: MetricScores({"mae": 1.0})


from ml_toolkit.data_analysis import DataAnalyzer, NumericalFeatureDistributions


def wrong_report_kind(data: Pool) -> None:
    train_model(splitter=Holdout(data), params={}, model_factory=CatBoostRegressionModel, metric=metrics, run_name="wrong-kind", config=TrainingConfig(), data_analyzers=PredictionEvaluator(name="prediction"))  # expected-type-error
    train_model(splitter=Holdout(data), params={}, model_factory=CatBoostRegressionModel, metric=metrics, run_name="wrong-kind", config=TrainingConfig(), prediction_evaluators=NumericalFeatureDistributions())  # expected-type-error
    DataAnalyzer(name="data").evaluate(data=data, split=pl.Series(), prediction=data.y.to_numpy())  # expected-type-error


TrainingConfig(fit_on_test="yes")  # expected-type-error
ReportingConfig(on_test=None)  # expected-type-error


def invalid_time_gap(data: Pool) -> None:
    Holdout(data, gap="7 days")  # expected-type-error


class WrongFullPool(WrongSplitter):
    def get_final(self) -> Split[Pool]:  # expected-type-error
        return Split(self.data)
