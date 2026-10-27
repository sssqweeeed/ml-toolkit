from ml_toolkit.training import Holdout, Split, Splitter, TimeSeriesCV
from dataclasses import replace
from ml_toolkit.feature_selection import (
    RFESelector,
    TopKSelector,
    ForwardSelector,
    FeatureSelector,
    FeatureSelectionResult,
)
from examples.custom_selector import FixedFeatureSelector
from ml_toolkit.types import TimePeriod
from datetime import date, timedelta
from pathlib import Path
from collections.abc import Callable, Iterator
from typing import assert_type

import polars as pl
import pandas as pd

from examples.project_pool import ProjectPool
from examples.custom_model import model_factory
from ml_toolkit import Pool, TrainingResult, train_model
from ml_toolkit.config import ReportingConfig, TrainingConfig
from ml_toolkit.metrics import (
    Metric,
    MetricScores,
    MetricsFrequency,
    MetricsRegression,
    metrics_regression,
    MetricsSeverity,
    PUMetric,
    metrics_classification,
    metrics_frequency,
    metrics_severity,
    ranking_metrics,
)
from ml_toolkit.models import CatBoostPUClassificationModel, CatBoostRegressionModel
from ml_toolkit.models.base import Model
from ml_toolkit.inflation import ConstantRateInflation, LogLinearInflation, SplineInflation
from ml_toolkit.evaluation import PredictionEvaluator
from ml_toolkit.config import FigureConfig, FeatureLevelConfig
from ml_toolkit.training import Pool as TrainingPool

from ml_toolkit.types import FloatArray, ModelParams


class BusinessEvaluator(PredictionEvaluator):
    def evaluate(
        self,
        *,
        data: Pool,
        prediction: FloatArray,
        split: pl.Series,
    ) -> dict[str, float]:
        return {"rows": float(len(data))}

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
        return []


def dictionary_metric(
    y: FloatArray, predictions: FloatArray, weight: FloatArray
) -> MetricScores:
    return MetricScores({"mae": float(abs(y - predictions).mean())})


def severity_metric(
    y: FloatArray, predictions: FloatArray, weight: FloatArray
) -> MetricsSeverity:
    return metrics_severity(y, predictions, weight, power=1.5)


def metric_contracts(y: FloatArray, prediction: FloatArray, weight: FloatArray) -> None:
    assert_type(metrics_frequency(y, prediction, weight), MetricsFrequency)
    assert_type(ranking_metrics(y, prediction, weight).gains_area, float)


def contracts(data: Pool) -> None:
    assert_type(ConstantRateInflation(0.03).fit(data), ConstantRateInflation)
    assert_type(LogLinearInflation(control_features=[]).fit(data), LogLinearInflation)
    assert_type(SplineInflation(n_knots=3).fit(data), SplineInflation)
    TrainingConfig(verbosity="silent")
    TrainingConfig(verbosity="normal")
    TrainingConfig(verbosity="verbose")
    TrainingConfig(verbosity="debug")
    assert_type(data.to_model_input(), pd.DataFrame)
    assert_type(data.to_model_input(5, categorical_as_strings=True), pd.DataFrame)
    reporting = ReportingConfig(on_tuning=True)
    assert_type(reporting.on_tuning, bool)
    ReportingConfig(on_validation=False, on_test=False, on_final_fit=False)
    evaluators: list[PredictionEvaluator] = [
        BusinessEvaluator(name="business"),
        BusinessEvaluator(name="quality"),
    ]
    assert_type(evaluators[0].name, str)
    inputs: tuple[PredictionEvaluator | list[PredictionEvaluator] | None, ...] = (
        None,
        evaluators[0],
        evaluators,
        [],
    )
    for evaluator in inputs:
        train_model(
            splitter=Holdout(data),
            params={},
            model_factory=CatBoostRegressionModel,
            metric=dictionary_metric,
            run_name="named-evaluators",
            config=TrainingConfig(),
            prediction_evaluators=evaluator,
        )
    assert_type(
        train_model(
            splitter=Holdout(data),
            params={},
            model_factory=CatBoostRegressionModel,
            metric=dictionary_metric,
            run_name="typing",
            config=TrainingConfig(),
        ),
        TrainingResult,
    )
    protocol: TrainingPool = data
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
    assert_type(list(Holdout(data).get_validation()), list[Split[Pool]])
    assert_type(
        list(TimeSeriesCV(data, n_splits=3).get_validation()),
        list[Split[Pool]],
    )
    assert_type(protocol.object_id, pl.Series | None)
    assert_type(model.get_refit_params(), dict[str, object])
    assert_type(
        metrics[0](
            data.y.to_numpy(), data.y.to_numpy(), data.weight.to_numpy()
        ).to_dict(),
        dict[str, float],
    )
    TrainingConfig()
    Holdout(data, test_size=4)
    Holdout(data, test_size=0.2)
    Holdout(data, test_size="2025-01-01")
    Holdout(data, test_size=date(2025, 1, 1))
    Holdout(data, test_size=None)


def project_pool_contract(data: ProjectPool) -> None:
    protocol: Pool = data
    training_protocol: TrainingPool = protocol
    contracts(training_protocol)
    assert_type(data.take([0]), ProjectPool)
    assert_type(data.select_features([data.X.columns[0]]), ProjectPool)
    assert_type(
        list(Holdout(data).get_validation()),
        list[Split[ProjectPool]],
    )
    assert_type(
        CatBoostRegressionModel({"iterations": 2}).fit(data),
        CatBoostRegressionModel,
    )


def custom_adapter_contract(data: Pool, params: ModelParams) -> None:
    factory: Callable[[ModelParams], Model] = model_factory
    adapter: Model = factory(params)
    adapter.fit(data)
    assert_type(adapter.predict_pool(data), FloatArray)
    assert_type(adapter.get_refit_params(), ModelParams)


from ml_toolkit.training import importance_ranker


def explicit_selection(data: Pool) -> None:
    settings = RFESelector(
        ranker=importance_ranker, min_features=1, metric="mae", direction="minimize"
    )
    assert_type(
        replace(settings, ranker=importance_ranker).select(
            train=data,
            validation=data,
            params={},
            model_factory=CatBoostRegressionModel,
            metric=dictionary_metric,
        ),
        FeatureSelectionResult,
    )


from ml_toolkit.config import FeatureLevelConfig
from ml_toolkit.feature_levels import FeatureLevels, LevelTransformer
from ml_toolkit.psi import PSIBaseline, PSIConfig, PSIResult
from ml_toolkit.tracking import load_psi_baseline


def monitoring_contract(
    reference: pl.DataFrame, current: pl.DataFrame, uri: str
) -> None:
    config = PSIConfig((FeatureLevelConfig("age", "numerical"),))
    baseline = PSIBaseline.from_reference(reference, config=config)
    assert_type(baseline, PSIBaseline)
    assert_type(baseline.compare(current), PSIResult)
    assert_type(baseline.transformer.transform(current), pl.DataFrame)
    assert_type(load_psi_baseline(uri), PSIBaseline)
    assert_type(PSIBaseline.load("baseline"), PSIBaseline)
    assert_type(
        LevelTransformer.from_dict(baseline.transformer.to_dict()), LevelTransformer
    )


from ml_toolkit.plots.regression import observed_fitted_by_level
from ml_toolkit.psi import compare_psi
import pandas as pd


def feature_diagnostics(
    reference: pl.DataFrame, current: pl.DataFrame, y: FloatArray, output: Path
) -> None:
    common = ReportingConfig(
        features=(FeatureLevelConfig("age", "numerical", lower_bound=0),),
        psi=False,
    )
    assert_type(common.features, tuple[FeatureLevelConfig, ...] | None)
    transformer = LevelTransformer((FeatureLevelConfig("age", "numerical"),)).fit(
        reference
    )
    assert_type(transformer.feature_levels, tuple[FeatureLevels, ...] | None)
    reference_levels = transformer.transform(reference)
    current_levels = transformer.transform(current)
    assert transformer.feature_levels is not None
    assert_type(transformer.feature_levels[0].config, FeatureLevelConfig)
    result = compare_psi(
        reference_levels, current_levels, feature_levels=transformer.feature_levels
    )
    assert_type(result.feature_levels, tuple[FeatureLevels, ...])
    assert_type(
        compare_psi(reference_levels, current_levels, transformer.feature_levels),
        PSIResult,
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
            level_order=transformer.feature_levels[0].levels,
        ),
        pd.DataFrame,
    )


def selector_contracts(data: Pool) -> None:
    selectors: tuple[FeatureSelector, ...] = (
        RFESelector(
            min_features=1, metric="mae", direction="minimize", ranker=importance_ranker
        ),
        TopKSelector(k=1, ranker=importance_ranker),
        ForwardSelector(
            metric="mae", direction="minimize", max_features=1, max_fits=20
        ),
        FixedFeatureSelector(("age",)),
    )
    for selector in selectors:
        assert_type(
            selector.select(
                train=data,
                validation=data,
                params={},
                model_factory=CatBoostRegressionModel,
                metric=dictionary_metric,
            ),
            FeatureSelectionResult,
        )
    period: TimePeriod = "hour"
    assert_type(ReportingConfig(period=period).period, TimePeriod)


class ProjectSplitter(Splitter[Pool]):
    def get_validation(self) -> Iterator[Split[Pool]]:
        yield Split(self.data.take([0]), self.data.take([1]), index=1)

    def get_test(self) -> Split[Pool]:
        return Split(self.data)


def splitter_contracts(data: ProjectPool) -> None:
    splitter: Splitter[Pool] = Holdout(data)
    assert_type(splitter.get_validation(), Iterator[Split[Pool]])
    assert_type(Holdout(data).get_test(), Split[ProjectPool])
    assert_type(
        TimeSeriesCV(data, n_splits=2).get_validation(), Iterator[Split[ProjectPool]]
    )
    assert_type(ProjectSplitter(data).get_test().heldout, Pool | None)


def new_training_contracts(data: Pool, result: TrainingResult, y: FloatArray) -> None:
    assert_type(Holdout(data, gap=timedelta(days=2)).get_final(), Pool)
    assert_type(
        TimeSeriesCV(data, n_splits=2, gap=timedelta(hours=3)).get_test(), Split[Pool]
    )
    assert_type(result.final_model, Model | None)
    assert_type(TrainingConfig(fit_on_test=True).fit_on_test, bool)
    metric: Metric = metrics_regression
    assert_type(metrics_regression(y, y), MetricsRegression)
    assert_type(metric(y, y, y).to_dict(), dict[str, float])


from ml_toolkit.data_analysis import DataAnalyzer, NumericalFeatureDistributions
from examples.custom_data_analyzer import DataAudit


def data_analysis_contracts(data: Pool, split: pl.Series, output: Path) -> None:
    analyzer: DataAnalyzer = DataAudit(name="audit")
    assert_type(analyzer.evaluate(data=data, split=split), dict[str, float])
    assert_type(
        NumericalFeatureDistributions().write_artifacts(
            data=data,
            split=split,
            artifact_dir=output,
            figure=FigureConfig(),
            features=None,
            n_bins=10,
        ),
        list[Path],
    )
    choices: tuple[DataAnalyzer | list[DataAnalyzer] | None, ...] = (
        None,
        analyzer,
        [],
        [analyzer, NumericalFeatureDistributions()],
    )
    for analyzers in choices:
        train_model(
            splitter=Holdout(data),
            params={},
            model_factory=CatBoostRegressionModel,
            metric=dictionary_metric,
            run_name="data-analysis",
            config=TrainingConfig(),
            data_analyzers=analyzers,
            prediction_evaluators=BusinessEvaluator(name="prediction"),
        )
