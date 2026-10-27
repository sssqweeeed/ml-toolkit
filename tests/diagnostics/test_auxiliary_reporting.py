"""Continuous auxiliary values retain reference levels and original distributions."""

from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ml_toolkit.config import FigureConfig, ReportingConfig
from ml_toolkit.data import Pool
from ml_toolkit.data_analysis import NumericalFeatureDistributions
from ml_toolkit.evaluation import ObservedFitted
from ml_toolkit.feature_levels import FeatureLevelConfig, LevelTransformer
from ml_toolkit.feature_levels.config import feature_definitions
from ml_toolkit.psi import PSIBaseline
from ml_toolkit.training.psi import prepare_psi_baseline
from ml_toolkit.training.reporting import _psi_artifacts


@pytest.fixture
def data():
    return Pool(
        pl.DataFrame({"x": np.arange(8, dtype=float)}),
        pl.Series(np.arange(8, dtype=float)),
        pl.Series([date(2025, 1, 1) + timedelta(days=i) for i in range(8)]),
        pl.Series([1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 4.0]),
        auxiliary_features=pl.DataFrame(
            {
                "amount": [10.0, 20.0, 30.0, 40.0, 10000.0, None, -1.0, 25.0],
                "code": [1, 2, 1, 2, 3, 1, 2, 1],
            }
        ),
    )


def test_observed_fitted_and_saved_psi_use_identical_train_only_levels(
    data, tmp_path, monkeypatch
):
    config = ReportingConfig(
        psi=True,
        n_bins=2,
        features=(
            FeatureLevelConfig("amount", "numerical", lower_bound=0),
            FeatureLevelConfig("code", "categorical"),
        ),
    )
    train, current = data.take(range(4)), data.take(range(4, 8))
    baseline = prepare_psi_baseline(train, config, log_model=True, stage="test")
    assert baseline.transformer.feature_levels[0].boundaries == (25.0,)
    baseline.save(tmp_path / "baseline")
    loaded = PSIBaseline.load(tmp_path / "baseline")
    assert loaded.transformer.feature_levels == baseline.transformer.feature_levels
    assert loaded.compare(train.to_analysis_frame()).psi["psi"].to_list() == [0.0, 0.0]
    with pytest.raises(ValueError, match="missing.*amount.*code"):
        loaded.compare(current.X)
    assert loaded.transformer.transform(current.to_analysis_frame())["amount"].cast(
        pl.String
    ).to_list() == [
        "[25, inf)",
        "__NULL__",
        "__ERROR__",
        "[25, inf)",
    ]
    assert (
        "__UNSEEN__"
        in loaded.transformer.transform(current.to_analysis_frame())["code"]
        .cast(pl.String)
        .to_list()
    )
    fits = []
    original = LevelTransformer.fit

    def capture(self, reference):
        result = original(self, reference)
        fits.append(self.feature_levels)
        return result

    monkeypatch.setattr(LevelTransformer, "fit", capture)
    split = pl.Series(["train"] * 4 + ["test"] * 4)
    paths = ObservedFitted().write_artifacts(
        data=data,
        prediction=np.ones(8),
        split=split,
        artifact_dir=tmp_path / "observed",
        figure=FigureConfig(),
        features=config.features,
        n_bins=config.n_bins,
    )
    assert len(paths) == 2 and all(path.is_file() for path in paths)
    assert fits == [baseline.transformer.feature_levels]
    _psi_artifacts(data, split, tmp_path / "psi", config, loaded)
    assert len(fits) == 1  # The saved transformer is reused.
    scores = pl.read_json(tmp_path / "psi/scores.json")
    assert_frame_equal(scores, loaded.compare(current.to_analysis_frame()).psi)
    assert_frame_equal(
        data.auxiliary_features,
        pl.concat([train.auxiliary_features, current.auxiliary_features]),
    )


def test_numerical_distribution_receives_original_values_and_infers_auxiliaries(
    data, tmp_path, monkeypatch
):
    captured = []
    monkeypatch.setattr(
        "ml_toolkit.data_analysis.monitoring.time_feature_distribution",
        lambda reference, current, name, *a, **kw: captured.append(
            (name, reference[name], current[name])
        ),
    )
    NumericalFeatureDistributions().write_artifacts(
        data=data,
        split=pl.Series(["train"] * 4 + ["test"] * 4),
        artifact_dir=tmp_path,
        figure=FigureConfig(),
        features=(),
        n_bins=1,
    )
    assert [name for name, _, _ in captured] == ["x", "amount", "code"]
    _, reference, current = captured[1]
    assert reference.to_list() == [10.0, 20.0, 30.0, 40.0]
    assert current.to_list() == [10000.0, None, -1.0, 25.0]
    definitions = feature_definitions(
        data, (FeatureLevelConfig("code", "categorical"),)
    )
    assert definitions[0].ftype == "categorical"


@pytest.mark.parametrize(
    "reference,boundaries",
    [([1.0, 1.0, 1.0], ()), ([None, None, None], ()), ([1.0, 1.0, 2.0, 2.0], (1.5,))],
)
def test_constant_missing_and_tied_references(reference, boundaries):
    transformer = LevelTransformer(
        (FeatureLevelConfig("value", "numerical"),), n_bins=2
    ).fit(pl.DataFrame({"value": reference}, schema={"value": pl.Float64}))
    assert transformer.feature_levels[0].boundaries == boundaries
    assert (
        transformer.transform(pl.DataFrame({"value": [0.0, 1.0, 100.0, None]})).height
        == 4
    )


@pytest.mark.parametrize("infinity", [float("inf"), float("-inf")])
def test_infinite_regular_values_rejected_on_fit_and_transform(infinity):
    definitions = (FeatureLevelConfig("value", "numerical", lower_bound=0),)
    with pytest.raises(ValueError, match="finite"):
        LevelTransformer(definitions).fit(pl.DataFrame({"value": [infinity]}))
    transformer = LevelTransformer(definitions).fit(pl.DataFrame({"value": [1.0, 2.0]}))
    with pytest.raises(ValueError, match="finite"):
        transformer.transform(pl.DataFrame({"value": [infinity]}))
    sentinel = LevelTransformer(
        (FeatureLevelConfig("value", "numerical", error_level=infinity),)
    ).fit(pl.DataFrame({"value": [1.0, infinity]}))
    assert (
        sentinel.transform(pl.DataFrame({"value": [infinity]}))["value"]
        .cast(pl.String)
        .item()
        == "__ERROR__"
    )


def test_auxiliary_features_survive_selection_reporting_and_final_refit(tmp_path):
    import mlflow

    from examples.custom_selector import FixedFeatureSelector
    from ml_toolkit.config import TrackingConfig, TrainingConfig
    from ml_toolkit.evaluation import GroupedMetrics, TimeObservedFitted
    from ml_toolkit.metrics import metrics_regression
    from ml_toolkit.training import Holdout, train_model
    from tests.support.training import (
        MeanModel,
        list_all_artifacts,
        mae,
        make_identified_data,
        setup_mlflow,
    )

    setup_mlflow(tmp_path)
    data = make_identified_data()
    data = replace(
        data,
        auxiliary_features=pl.DataFrame(
            {
                "segment": ["a", "b"] * 12,
                "amount": np.arange(24, dtype=float) * 100,
            }
        ),
    )
    baselines = []

    class Model(MeanModel):
        def fit(self, train, validation=None):
            self.training = train
            return super().fit(train, validation)

        def log(self, sample, *, config=None, extra_artifacts=None):
            assert sample.X.columns == ["x"]
            assert sample.to_model_input().columns.to_list() == ["x"]
            baseline = PSIBaseline.load(extra_artifacts["psi_baseline"])
            assert [item.name for item in baseline.transformer.features] == [
                "x",
                "segment",
                "amount",
            ]
            assert baseline.compare(self.training.to_analysis_frame()).psi[
                "psi"
            ].to_list() == [0.0, 0.0, 0.0]
            baselines.append(baseline)

    result = train_model(
        splitter=Holdout(data, val_size=4, test_size=4),
        params={},
        model_factory=Model,
        metric=mae,
        run_name="auxiliary-workflow",
        feature_selector=FixedFeatureSelector(("x",)),
        prediction_evaluators=[
            ObservedFitted(features=("amount",)),
            TimeObservedFitted(by=("segment",)),
            GroupedMetrics(metric=metrics_regression, by=("segment",)),
        ],
        data_analyzers=NumericalFeatureDistributions(features=("amount",)),
        config=TrainingConfig(
            fit_on_test=True,
            reporting=ReportingConfig(psi=True),
            tracking=TrackingConfig(log_validation_models=True),
        ),
    )
    assert result.features == ("x",)
    assert [item.metadata["n_rows"] for item in baselines] == [16, 20, 24]
    client = mlflow.MlflowClient()
    stages = set()
    for run in mlflow.search_runs(output_format="list"):
        name = run.data.tags["mlflow.runName"]
        if name not in {"validation/split_1", "test", "final_fit"}:
            continue
        stages.add(name)
        artifacts = list_all_artifacts(client, run.info.run_id)
        assert "grouped_metrics/metrics.parquet" in artifacts
        assert any(key.startswith("grouped_metrics/") for key in run.data.metrics)
    assert stages == {"validation/split_1", "test", "final_fit"}
    assert set(result.test_metrics) == {"mae"}
