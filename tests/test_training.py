"""Train with a project-owned data object and inspect MLflow artifacts."""

from dataclasses import dataclass
from datetime import date, timedelta

import mlflow
import numpy as np
import polars as pl
import pytest

from ml_toolkit.models import CatBoostRegressionModel
from ml_toolkit.models import CatBoostClassificationModel
from ml_toolkit.metrics import PUMetric, metrics_classification
from ml_toolkit.inflation import ConstantRateInflation
from ml_toolkit.models import InflationAdjustedModel
from ml_toolkit.explainability import calculate_shap_values, top_shap_features
from ml_toolkit.plots.explainability import shap_beeswarm, shap_dependence, shap_importance
from ml_toolkit.training import DiagnosticsOptions, train_model


@dataclass(frozen=True)
class ProjectData:
    X: pl.DataFrame
    y: pl.Series
    time: pl.Series
    weight: pl.Series
    cat_features: tuple[str, ...] = ()

    def __len__(self):
        return len(self.X)

    def take(self, indices):
        index = np.asarray(indices, dtype=np.int64)
        return ProjectData(self.X[index], self.y.gather(index), self.time.gather(index),
                           self.weight.gather(index), self.cat_features)

    def select_features(self, names):
        return ProjectData(self.X.select(names), self.y, self.time, self.weight,
                           tuple(name for name in self.cat_features if name in names))

    def with_target(self, target):
        values = target if isinstance(target, pl.Series) else pl.Series("target", target)
        return ProjectData(self.X, values, self.time, self.weight, self.cat_features)

    def to_catboost(self, baseline=None):
        import catboost

        baseline_values = None if baseline is None else np.full(len(self), baseline)
        return catboost.Pool(self.X.to_pandas(), label=self.y.to_numpy(),
                             weight=self.weight.to_numpy(),
                             cat_features=list(self.cat_features), baseline=baseline_values)

    def to_model_input(self, n_rows=None):
        frame = self.X if n_rows is None else self.X.head(n_rows)
        return frame.to_pandas()


class MeanModel:
    def __init__(self, params):
        self.params = params
        self.mean = None

    @property
    def best_iteration(self):
        return None

    def fit(self, train, validation=None):
        self.mean = float(np.average(train.y.to_numpy(), weights=train.weight.to_numpy()))
        return self

    def predict_pool(self, data):
        return np.full(len(data), self.mean)

    def select_features(self, train, validation, num_features_to_select):
        return train.X.columns[:num_features_to_select]

    def log(self, sample):
        mlflow.log_dict({"mean": self.mean}, "model.json")


def mae(y, prediction, weight):
    return {"mae": np.average(np.abs(y - prediction), weights=weight)}


def make_data():
    length = 24
    return ProjectData(
        X=pl.DataFrame({"x": np.arange(length, dtype=float), "unused": np.arange(length, dtype=float) % 3}),
        y=pl.Series("target", np.arange(length, dtype=float) + 1),
        time=pl.Series("date", [date(2024, 1, 1) + timedelta(days=30 * n) for n in range(length)]),
        weight=pl.Series("weight", np.ones(length)),
    )


def setup_mlflow(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("toolkit-test")


def list_all_artifacts(client, run_id, path=""):
    result = []
    for item in client.list_artifacts(run_id, path):
        result.extend(list_all_artifacts(client, run_id, item.path) if item.is_dir else [item.path])
    return result


def test_train_model_selection_and_mlflow(tmp_path):
    setup_mlflow(tmp_path)
    result = train_model(
        data=make_data(), params={"unused_param": 1}, model_factory=MeanModel,
        metric=mae, run_name="example", n_splits=2, select_features=1,
    )
    assert result.features == ("x",)
    assert len(result.cross_validation.fold_metrics) == 2
    assert "mae" in result.cross_validation.mean_metrics
    runs = mlflow.search_runs(output_format="list")
    assert any(run.data.tags.get("mlflow.runName") == "final_fit" for run in runs)
    parent = next(run for run in runs if run.data.tags.get("mlflow.runName") == "example")
    assert "selection/features.json" in list_all_artifacts(mlflow.MlflowClient(), parent.info.run_id)


def test_training_rejects_unsorted_project_data(tmp_path):
    setup_mlflow(tmp_path)
    source = make_data()
    reversed_rows = source.take(np.arange(len(source))[::-1])
    with pytest.raises(ValueError, match="ascending timestamps"):
        train_model(data=reversed_rows, params={}, model_factory=MeanModel,
                    metric=mae, run_name="invalid", n_splits=2)


def test_training_logs_psi_and_plots(tmp_path):
    setup_mlflow(tmp_path)
    train_model(
        data=make_data(), params={}, model_factory=MeanModel, metric=mae,
        run_name="diagnostics", n_splits=2,
        diagnostics=DiagnosticsOptions(task="regression", psi_bins=3,
                                       feature_plots=("x",)),
    )
    runs = mlflow.search_runs(output_format="list")
    fold = next(run for run in runs if run.data.tags.get("mlflow.runName") == "fold_1")
    artifacts = list_all_artifacts(mlflow.MlflowClient(), fold.info.run_id)
    assert "psi/fold_1/scores.json" in artifacts
    assert any(path.endswith("psi_scores.png") for path in artifacts)
    assert any(path.endswith("monthly_observed_fitted.png") for path in artifacts)


def test_bundled_catboost_model_runs_through_training(tmp_path):
    setup_mlflow(tmp_path)
    result = train_model(
        data=make_data(), model_factory=CatBoostRegressionModel,
        params={"iterations": 5, "loss_function": "Poisson", "depth": 2,
                "learning_rate": 0.1, "verbose": False},
        metric=mae, run_name="catboost", n_splits=2,
    )
    prediction = result.model.predict_pool(make_data())
    assert prediction.shape == (24,)
    assert np.isfinite(prediction).all()
    runs = mlflow.search_runs(output_format="list")
    final = next(run for run in runs if run.data.tags.get("mlflow.runName") == "final_fit")
    loaded = mlflow.pyfunc.load_model(f"runs:/{final.info.run_id}/catboost_model")
    loaded_prediction = np.asarray(loaded.predict(make_data().to_model_input()))
    assert np.allclose(loaded_prediction, prediction)


def test_bundled_classifier_and_classification_diagnostics(tmp_path):
    setup_mlflow(tmp_path)
    source = make_data()
    binary = ProjectData(source.X, pl.Series("target", [0, 1] * 12),
                         source.time, source.weight)
    result = train_model(
        data=binary, model_factory=CatBoostClassificationModel,
        params={"iterations": 5, "loss_function": "Logloss", "depth": 2,
                "learning_rate": 0.1, "verbose": False},
        metric=metrics_classification, run_name="classification", n_splits=2,
        diagnostics=DiagnosticsOptions(task="classification", psi_bins=3,
                                       feature_plots=()),
    )
    assert result.model.predict_pool(binary).shape == (24,)
    runs = mlflow.search_runs(output_format="list")
    fold = next(run for run in runs if run.data.tags.get("mlflow.runName") == "fold_1")
    artifacts = list_all_artifacts(mlflow.MlflowClient(), fold.info.run_id)
    assert any(path.endswith("precision_recall.png") for path in artifacts)
    assert any(path.endswith("calibration.png") for path in artifacts)
    final = next(run for run in runs if run.data.tags.get("mlflow.runName") == "final_fit")
    loaded = mlflow.pyfunc.load_model(f"runs:/{final.info.run_id}/model")
    loaded_prediction = np.asarray(loaded.predict(binary.to_model_input()))
    assert np.allclose(loaded_prediction, result.model.predict_pool(binary))


def test_optuna_tuning_uses_same_training_function(tmp_path):
    setup_mlflow(tmp_path)
    result = train_model(
        data=make_data(), params={}, model_factory=MeanModel, metric=mae,
        run_name="tuning", n_splits=2, tune_params=True,
        param_space=lambda trial: {"unused_param": trial.suggest_int("unused_param", 1, 2)},
        metric_to_optimize="mae", optimize_direction="minimize", n_trials=2,
    )
    assert result.params["unused_param"] in (1, 2)
    assert len(result.cross_validation.fold_metrics) == 2


def test_inflation_model_and_shap_are_available(tmp_path):
    data = make_data()
    regression = CatBoostRegressionModel({"iterations": 5, "loss_function": "Poisson",
                                          "depth": 2, "verbose": False})
    adjusted = InflationAdjustedModel(regression, ConstantRateInflation(.03), date(2024, 1, 1))
    adjusted.fit(data)
    assert np.isfinite(adjusted.predict_pool(data)).all()

    explanation = calculate_shap_values(regression.native_model, data.X, n_samples=10)
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
    binary = ProjectData(source.X, pl.Series("target", [0, 1] * 12),
                         source.time, source.weight)
    model = CatBoostClassificationModel({
        "loss": "uPU", "pu_prior": 0.5, "iterations": 5, "depth": 2,
        "learning_rate": 0.1, "thread_count": 1,
        "allow_writing_files": False, "logging_level": "Silent",
    }).fit(binary.take(np.arange(16)), binary.take(np.arange(16, 24)))
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
    binary = ProjectData(source.X, pl.Series("target", [0, 1] * 12),
                         source.time, source.weight)
    params = {"loss": "uPU", "pu_prior": 0.5, "iterations": 3, "depth": 2,
              "learning_rate": 0.1, "thread_count": 1,
              "allow_writing_files": False, "logging_level": "Silent"}
    result = train_model(
        data=binary, model_factory=CatBoostClassificationModel, params=params,
        metric=PUMetric(0.5, loss="uPU"), run_name="pu", n_splits=2,
        diagnostics=DiagnosticsOptions(task="pu_classification", psi=False,
                                       feature_plots=()),
    )
    assert "pu_risk" in result.cross_validation.mean_metrics
    runs = mlflow.search_runs(output_format="list")
    fold = next(run for run in runs if run.data.tags.get("mlflow.runName") == "fold_1")
    artifacts = list_all_artifacts(mlflow.MlflowClient(), fold.info.run_id)
    assert any(path.endswith("pu_score_distribution.png") for path in artifacts)
    assert any(path.endswith("pu_positive_capture.png") for path in artifacts)
    assert not any(path.endswith(("precision_recall.png", "calibration.png", "gains.png"))
                   for path in artifacts)


def test_pu_training_rejects_supervised_metrics(tmp_path):
    setup_mlflow(tmp_path)
    source = make_data()
    binary = ProjectData(source.X, pl.Series("target", [0, 1] * 12),
                         source.time, source.weight)
    with pytest.raises(ValueError, match="require PUMetric"):
        train_model(
            data=binary, model_factory=CatBoostClassificationModel,
            params={"loss": "uPU", "pu_prior": 0.5, "iterations": 2, "depth": 2,
                    "thread_count": 1, "allow_writing_files": False,
                    "logging_level": "Silent"},
            metric=metrics_classification, run_name="invalid_pu", n_splits=2,
        )
