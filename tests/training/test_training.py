"""Train with a project-owned data object and inspect MLflow artifacts."""

import json
import logging
from datetime import date
from functools import partial
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
import pytest

from ml_toolkit.feature_selection import RFESelector
from ml_toolkit.config import (
    ClassificationDiagnosticsConfig,
    ClassificationPlotsConfig,
    CommonPlotsConfig,
    NumericalFeatureDistributionConfig,
    PUDiagnosticsConfig,
    PUPlotsConfig,
    RegressionDiagnosticsConfig,
    RegressionPlotsConfig,
    ReportingConfig,
    ShapConfig,
    TimeSeriesCVConfig,
    TrainingConfig,
)
from ml_toolkit.explainability import calculate_shap_values, top_shap_features
from ml_toolkit.inflation import ConstantRateInflation
from ml_toolkit.metrics import PUMetric, metrics_classification, metrics_severity
from ml_toolkit.models import (
    CatBoostClassificationModel,
    CatBoostLogLinkRegressionModel,
    CatBoostPUClassificationModel,
    CatBoostRegressionModel,
    InflationAdjustedModel,
)
from ml_toolkit.plots.explainability import (
    shap_beeswarm,
    shap_dependence,
    shap_importance,
)
from ml_toolkit.training import importance_ranker, train_model


from tests.support.training import (
    ProjectData,
    MeanModel,
    mae,
    make_data,
    setup_mlflow,
    list_all_artifacts,
)


def recording_model():
    calls = {"selection": [], "fit": []}

    class RecordingModel(MeanModel):
        def fit(self, train, validation=None):
            calls["fit"].append(tuple(train.X.columns))
            if validation is not None:
                assert tuple(validation.X.columns) == tuple(train.X.columns)
            super().fit(train, validation)
            if train.X.columns == ["x", "unused"] and validation is not None:
                calls["selection"].append(
                    (tuple(train.X["x"].to_list()), tuple(validation.X["x"].to_list()))
                )
                self._native.feature_importances_ = np.array([0.0, 1.0])
            return self

    return RecordingModel, calls


def test_train_model_selection_and_mlflow(tmp_path):
    setup_mlflow(tmp_path)
    result = train_model(
        data=make_data(),
        params={"unused_param": 1},
        model_factory=MeanModel,
        metric=mae,
        run_name="example",
        config=TrainingConfig(validation=TimeSeriesCVConfig(n_splits=2)),
        feature_selector=RFESelector(
            ranker=importance_ranker, min_features=1, metric="mae", direction="minimize"
        ),
    )
    assert result.features == ("x",)
    assert len(result.validation.splits) == 2
    assert "mae" in result.validation.mean_metrics
    runs = mlflow.search_runs(output_format="list")
    assert any(run.data.tags.get("mlflow.runName") == "final_fit" for run in runs)
    last = next(
        run for run in runs if run.data.tags["mlflow.runName"] == "validation/split_2"
    )
    artifact = mlflow.artifacts.download_artifacts(
        run_id=last.info.run_id, artifact_path="selection/history.json"
    )
    history = json.loads(Path(artifact).read_text())
    assert history["features"] == list(result.features)
    assert history["selected_step"] == result.feature_selection.selected_step
    assert [step["features"] for step in history["steps"]] == [["x", "unused"], ["x"]]
    assert last.data.metrics["selection/n_fits"] == 2
    assert last.data.metrics["selection/n_features"] == 1


@pytest.mark.parametrize("skip_splits", [0, 1])
def test_feature_selection_uses_each_evaluated_split_training_rows(
    tmp_path,
    skip_splits,
):
    setup_mlflow(tmp_path)
    model_factory, calls = recording_model()
    result = train_model(
        data=make_data(),
        params={},
        model_factory=model_factory,
        metric=mae,
        run_name="early-selection",
        config=TrainingConfig(
            validation=TimeSeriesCVConfig(n_splits=2, skip_splits=skip_splits)
        ),
        feature_selector=RFESelector(
            ranker=importance_ranker, min_features=1, metric="mae", direction="minimize"
        ),
    )

    expected_rows = [8, 16][skip_splits:]
    assert calls["selection"] == [
        (tuple(range(n)), tuple(range(n, n + 8))) for n in expected_rows
    ]
    assert calls["fit"] == [("x", "unused"), ("unused",), ("unused",)] * len(
        expected_rows
    ) + [("unused",)]
    assert result.features == ("unused",)


def test_training_without_feature_selection_keeps_all_features(tmp_path):
    setup_mlflow(tmp_path)
    model_factory, calls = recording_model()
    result = train_model(
        data=make_data(),
        params={},
        model_factory=model_factory,
        metric=mae,
        run_name="no-selection",
        config=TrainingConfig(validation=TimeSeriesCVConfig(n_splits=2)),
    )

    assert calls["fit"] == [("x", "unused")] * 3
    assert result.features == ("x", "unused")


def test_feature_selection_accepts_single_row_training_window(tmp_path):
    setup_mlflow(tmp_path)
    result = train_model(
        data=make_data().take(np.arange(3)),
        params={},
        model_factory=MeanModel,
        metric=mae,
        run_name="small-selection-window",
        config=TrainingConfig(validation=TimeSeriesCVConfig(n_splits=2)),
        feature_selector=RFESelector(
            ranker=importance_ranker, min_features=1, metric="mae", direction="minimize"
        ),
    )
    assert len(result.validation.splits) == 2
    assert all(
        len(part.feature_selection.steps) == 2 for part in result.validation.splits
    )


def test_training_rejects_unsorted_project_data(tmp_path):
    setup_mlflow(tmp_path)
    source = make_data()
    reversed_rows = source.take(np.arange(len(source))[::-1])
    with pytest.raises(ValueError, match="ascending timestamps"):
        train_model(
            data=reversed_rows,
            params={},
            model_factory=MeanModel,
            metric=mae,
            run_name="invalid",
            config=TrainingConfig(validation=TimeSeriesCVConfig(n_splits=2)),
        )


def test_training_logs_psi_and_plots(tmp_path, caplog):
    setup_mlflow(tmp_path)
    caplog.set_level(logging.DEBUG, logger="ml_toolkit")
    train_model(
        data=make_data(),
        params={},
        model_factory=MeanModel,
        metric=mae,
        run_name="diagnostics",
        config=TrainingConfig(
            validation=TimeSeriesCVConfig(n_splits=2),
            reporting=ReportingConfig(
                diagnostics=RegressionDiagnosticsConfig(
                    task="regression",
                    plots=RegressionPlotsConfig(
                        common=CommonPlotsConfig(
                            n_bins=3,
                            numerical_distributions=NumericalFeatureDistributionConfig(
                                features=("x",)
                            ),
                        ),
                    ),
                ),
                validation_splits="all",
            ),
        ),
    )
    runs = mlflow.search_runs(output_format="list")
    folds = [
        run
        for run in runs
        if run.data.tags.get("mlflow.runName", "").startswith("validation/split_")
    ]
    assert {run.data.tags["mlflow.runName"] for run in folds} == {
        "validation/split_1",
        "validation/split_2",
    }
    for fold in folds:
        artifacts = list_all_artifacts(mlflow.MlflowClient(), fold.info.run_id)
        assert set(artifacts) == {
            "tables/psi/scores.json",
            "tables/psi/level_distribution.json",
            "tables/psi/time_distribution.json",
            "figures/psi/scores.png",
            "figures/psi/time_by_feature/000_x.png",
            "figures/psi/time_by_feature/001_unused.png",
            "figures/observed_fitted/time.png",
            "figures/observed_fitted/by_feature/000_x.png",
            "figures/observed_fitted/by_feature/001_unused.png",
            "figures/feature_distributions/time_by_feature/000_x.png",
        }
    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "ml_toolkit.training.reporting"
    ]
    assert any(
        message.startswith("Generating diagnostics for validation/split_1")
        for message in messages
    )
    assert (
        "Logged diagnostic figure figures/psi/scores.png for validation/split_1"
        in messages
    )


@pytest.mark.parametrize("loss", ["Poisson", "Gamma"])
def test_bundled_catboost_model_runs_through_training(tmp_path, loss):
    setup_mlflow(tmp_path)
    result = train_model(
        data=make_data(),
        model_factory=(
            CatBoostLogLinkRegressionModel
            if loss == "Gamma"
            else CatBoostRegressionModel
        ),
        params={
            "iterations": 5,
            "thread_count": 1,
            "allow_writing_files": False,
            "loss_function": loss,
            "depth": 2,
            "learning_rate": 0.1,
            "verbose": False,
        },
        metric=partial(metrics_severity, power=2.0) if loss == "Gamma" else mae,
        run_name="catboost",
        config=TrainingConfig(validation=TimeSeriesCVConfig(n_splits=2)),
    )
    prediction = result.model.predict_pool(make_data())
    assert prediction.shape == (24,)
    assert np.isfinite(prediction).all()
    runs = mlflow.search_runs(output_format="list")
    final = next(
        run for run in runs if run.data.tags.get("mlflow.runName") == "final_fit"
    )
    loaded = mlflow.pyfunc.load_model(f"runs:/{final.info.run_id}/model")
    loaded_prediction = np.asarray(loaded.predict(make_data().to_model_input()))
    assert np.allclose(loaded_prediction, prediction)


def test_bundled_classifier_and_classification_diagnostics(tmp_path):
    setup_mlflow(tmp_path)
    source = make_data()
    binary = ProjectData(
        source.X, pl.Series("target", [0, 1] * 12), source.time, source.weight
    )
    result = train_model(
        data=binary,
        model_factory=CatBoostClassificationModel,
        params={
            "iterations": 5,
            "thread_count": 1,
            "allow_writing_files": False,
            "loss_function": "Logloss",
            "depth": 2,
            "learning_rate": 0.1,
            "verbose": False,
        },
        metric=metrics_classification,
        run_name="classification",
        config=TrainingConfig(
            validation=TimeSeriesCVConfig(n_splits=2),
            reporting=ReportingConfig(
                diagnostics=ClassificationDiagnosticsConfig(
                    plots=ClassificationPlotsConfig(
                        common=CommonPlotsConfig(
                            n_bins=3,
                            numerical_distributions=NumericalFeatureDistributionConfig(
                                features=()
                            ),
                        ),
                    ),
                ),
                validation_splits="all",
            ),
        ),
    )
    assert result.model.predict_pool(binary).shape == (24,)
    runs = mlflow.search_runs(output_format="list")
    fold = next(
        run
        for run in runs
        if run.data.tags.get("mlflow.runName") == "validation/split_1"
    )
    artifacts = list_all_artifacts(mlflow.MlflowClient(), fold.info.run_id)
    assert "figures/performance/val/precision_recall.png" in artifacts
    assert "figures/performance/val/calibration.png" in artifacts
    for path in artifacts:
        if path.startswith("figures/performance/"):
            assert path.split("/")[2] == "val"
        else:
            assert path.split("/")[2] not in {"val", "test"}
    final = next(
        run for run in runs if run.data.tags.get("mlflow.runName") == "final_fit"
    )
    loaded = mlflow.pyfunc.load_model(f"runs:/{final.info.run_id}/model")
    loaded_prediction = np.asarray(loaded.predict(binary.to_model_input()))
    assert np.allclose(loaded_prediction, result.model.predict_pool(binary))


def test_inflation_model_and_shap_are_available(tmp_path, caplog):
    caplog.set_level(logging.DEBUG, logger="ml_toolkit")
    data = make_data()
    regression = CatBoostRegressionModel(
        {"iterations": 5, "loss_function": "Poisson", "depth": 2, "verbose": False,
         "thread_count": 1, "allow_writing_files": False}
    )
    adjusted = InflationAdjustedModel(
        regression, ConstantRateInflation(0.03), date(2024, 1, 1)
    )
    adjusted.fit(data)
    assert np.isfinite(adjusted.predict_pool(data)).all()
    assert any(
        record.name == "ml_toolkit.models.inflation_regression"
        and record.getMessage() == "Fitted inflation-adjusted model"
        for record in caplog.records
    )

    explanation = calculate_shap_values(
        regression.native_model, data.X, config=ShapConfig(max_samples=10)
    )
    assert explanation.values.shape == (10, 2)
    assert top_shap_features(explanation, 1)[0] in data.X.columns
    plot = tmp_path / "shap.png"
    shap_importance(explanation, plot)
    assert plot.is_file()
    shap_beeswarm(explanation, tmp_path / "beeswarm.png")
    shap_dependence(explanation, "x", tmp_path / "dependence.png")
    assert (tmp_path / "beeswarm.png").is_file()
    assert (tmp_path / "dependence.png").is_file()


def test_positive_unlabeled_classifier_is_preserved():
    source = make_data()
    binary = ProjectData(
        source.X, pl.Series("target", [0, 1] * 12), source.time, source.weight
    )
    model = CatBoostPUClassificationModel(
        {
            "loss": "uPU",
            "pu_prior": 0.5,
            "iterations": 5,
            "depth": 2,
            "learning_rate": 0.1,
            "thread_count": 1,
            "allow_writing_files": False,
            "logging_level": "Silent",
        }
    ).fit(binary.take(np.arange(16)), binary.take(np.arange(16, 24)))
    probability = model.predict_pool(binary)
    assert probability.shape == (24,)
    assert np.isfinite(probability).all()
    assert np.all((probability >= 0) & (probability <= 1))
    assert model.validation_f1 is None
    with pytest.raises(RuntimeError, match="decision_threshold"):
        model.predict_labels_pool(binary)


def test_pu_training_logs_only_pu_diagnostics(tmp_path):
    setup_mlflow(tmp_path)
    source = make_data()
    binary = ProjectData(
        source.X, pl.Series("target", [0, 1] * 12), source.time, source.weight
    )
    params = {
        "loss": "uPU",
        "pu_prior": 0.5,
        "iterations": 3,
        "depth": 2,
        "learning_rate": 0.1,
        "thread_count": 1,
        "allow_writing_files": False,
        "logging_level": "Silent",
    }
    result = train_model(
        data=binary,
        model_factory=CatBoostPUClassificationModel,
        params=params,
        metric=PUMetric(0.5, loss="uPU"),
        run_name="pu",
        config=TrainingConfig(
            validation=TimeSeriesCVConfig(n_splits=2),
            reporting=ReportingConfig(
                diagnostics=PUDiagnosticsConfig(
                    plots=PUPlotsConfig(
                        common=CommonPlotsConfig(
                            n_bins=3,
                            psi=False,
                            observed_fitted=False,
                            numerical_distributions=NumericalFeatureDistributionConfig(
                                features=()
                            ),
                        ),
                    ),
                ),
                validation_splits="all",
            ),
        ),
    )
    assert "pu_risk" in result.validation.mean_metrics
    runs = mlflow.search_runs(output_format="list")
    fold = next(
        run
        for run in runs
        if run.data.tags.get("mlflow.runName") == "validation/split_1"
    )
    artifacts = list_all_artifacts(mlflow.MlflowClient(), fold.info.run_id)
    assert "figures/performance/val/score_distribution.png" in artifacts
    assert "figures/performance/val/positive_capture.png" in artifacts
    assert all(path.split("/")[2] == "val" for path in artifacts)
    assert not any(
        path.endswith(("precision_recall.png", "calibration.png", "gains.png"))
        for path in artifacts
    )
