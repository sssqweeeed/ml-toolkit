"""Application-controlled logging and independent experiment tracking."""

import io
import json
import logging
import os
import runpy
import subprocess
import sys
from datetime import date
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
import polars as pl
import pytest
from test_training import MeanModel, mae, make_data, setup_mlflow

from ml_toolkit.feature_selection import RFESelector
from ml_toolkit.config import (
    HoldoutConfig,
    PostFitEvaluationConfig,
    ShapConfig,
    TimeSeriesCVConfig,
    TrainingConfig,
    TuningConfig,
)
from ml_toolkit.explainability import calculate_shap_values
from ml_toolkit.inflation import ConstantRateInflation
from ml_toolkit.metrics import MetricScores
from ml_toolkit.models import CatBoostClassificationModel, CatBoostRegressionModel
from ml_toolkit.plots.utils import save_figure
from ml_toolkit.feature_levels import FeatureLevelSpec
from ml_toolkit.psi import PSIAnalysis, PSIConfig
from ml_toolkit.tracking.mlflow import log_polars_table
from ml_toolkit.tracking.model_artifacts import log_model_from_code
from ml_toolkit.training import importance_ranker, split_holdout, train_model


def toolkit_records(caplog):
    return [
        record for record in caplog.records if record.name.startswith("ml_toolkit.")
    ]


@pytest.mark.parametrize("configured", [False, True])
def test_import_preserves_application_logging(tmp_path, configured):
    # A fresh process prevents pytest and prior imports from hiding side effects.
    code = """
import io
import logging
import sys

root = logging.getLogger()
package = logging.getLogger("ml_toolkit")
if sys.argv[1] == "True":
    handler = logging.StreamHandler(io.StringIO())
    handler.setLevel(logging.ERROR)
    handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    package.addHandler(logging.NullHandler())
    package.setLevel(logging.INFO)
    package.propagate = False
root_state = (root.level, tuple(root.handlers))
package_state = (package.level, package.propagate, tuple(package.handlers))
import ml_toolkit
assert (root.level, tuple(root.handlers)) == root_state
assert (package.level, package.propagate) == package_state[:2]
assert all(handler in package.handlers for handler in package_state[2])
added = [handler for handler in package.handlers if handler not in package_state[2]]
assert len(added) == 1 and type(added[0]) is logging.NullHandler
assert all(logging.getLogger(name).level == logging.NOTSET for name in (
    "ml_toolkit.training.workflow", "ml_toolkit.training.fit",
    "ml_toolkit.models.catboost_classification",
))
assert ml_toolkit.__all__ == ["Pool", "TrainingResult", "train_model"]
assert not any(name.startswith("ml_toolkit.training") for name in sys.modules)
from ml_toolkit import Pool, TrainingResult, train_model
from ml_toolkit.training.contracts import Pool as DirectPool
from ml_toolkit.training import TrainingResult as DirectResult, train_model as direct_train
assert Pool is DirectPool
assert TrainingResult is DirectResult
assert train_model is direct_train
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    result = subprocess.run(
        [sys.executable, "-c", code, str(configured)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("level", [logging.INFO, logging.DEBUG, logging.WARNING])
def test_workflow_levels_and_results(tmp_path, caplog, level):
    setup_mlflow(tmp_path)
    caplog.set_level(level, logger="ml_toolkit")
    optuna_verbosity = optuna.logging.get_verbosity()
    result = train_model(
        data=make_data(),
        params={"private_parameter": "do-not-log"},
        model_factory=MeanModel,
        metric=lambda y, p, w: MetricScores({"mae": float(np.mean(abs(y - p)))}),
        run_name="logging",
        param_space=lambda trial: {"choice": trial.suggest_int("choice", 1, 2)},
        config=TrainingConfig(
            validation=TimeSeriesCVConfig(n_splits=2, skip_splits=1),
            tuning=TuningConfig(metric="mae", direction="minimize", n_trials=2),
        ),
        feature_selector=RFESelector(
            ranker=importance_ranker, min_features=1, metric="mae", direction="minimize"
        ),
    )
    assert result.features == ("x",)
    assert optuna.logging.get_verbosity() == optuna_verbosity
    assert len(result.validation.splits) == 1
    assert result.model.predict_pool(make_data()).shape == (24,)
    records = toolkit_records(caplog)
    messages = [record.getMessage() for record in records]
    assert not any("do-not-log" in message for message in messages)
    if level == logging.WARNING:
        assert records == []
        return
    for milestone in (
        "Starting training run logging",
        "Completed feature selection: 1 features selected",
        "Starting Optuna tuning:",
        "Completed Optuna tuning:",
        "Starting validation split 2",
        "Completed validation split 2",
        "Completed validation",
        "Starting final fit",
        "Completed final fit",
        "Completed training run logging",
    ):
        assert any(message.startswith(milestone) for message in messages)
    debug_records = [record for record in records if record.levelno == logging.DEBUG]
    if level == logging.INFO:
        assert debug_records == []
    else:
        for detail in (
            "Training data:",
            "Selected features:",
            "Split 2:",
            "Starting tuning trial 1",
            "Completed tuning trial 2",
            "Evaluation validation/split_2 val metrics:",
            "Validation mean metrics:",
        ):
            assert any(message.startswith(detail) for message in messages)


def test_module_override_propagates_once(caplog):
    caplog.set_level(logging.INFO, logger="ml_toolkit")
    caplog.set_level(logging.DEBUG, logger="ml_toolkit.feature_selection")
    package = logging.getLogger("ml_toolkit")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    package.addHandler(handler)
    try:
        partition = split_holdout(make_data(), HoldoutConfig(val_size=0.25))
        selected = RFESelector(
            ranker=importance_ranker, min_features=1, metric="mae", direction="minimize"
        ).select(
            metric=mae,
            train=partition.train,
            validation=partition.val,
            params={},
            model_factory=MeanModel,
        )
    finally:
        package.removeHandler(handler)
        handler.close()
    assert list(selected.features) == ["x"]
    messages = [record.getMessage() for record in toolkit_records(caplog)]
    assert "Selected features: ('x',)" in messages
    assert stream.getvalue().splitlines() == messages
    assert (
        messages.count("Completed feature selection: 1 features selected at step 2")
        == 1
    )


def test_helper_debug_logs_preserve_outputs(tmp_path, caplog, monkeypatch):
    caplog.set_level(logging.DEBUG, logger="ml_toolkit")
    data = make_data()
    psi = PSIAnalysis(PSIConfig((FeatureLevelSpec("x", "numerical"),), n_bins=3))
    comparison = psi.fit(data.X).compare(data.X)
    assert comparison.psi["psi"].to_list() == [0.0]

    dates = pl.Series([date(2024, 1, 1)], dtype=pl.Date)
    adjusted = ConstantRateInflation(0.03).adjust(np.array([10.0]), dates, dates)
    np.testing.assert_allclose(adjusted, [10.0])

    fig, _ = plt.subplots()
    assert save_figure(fig, tmp_path / "figure.png").is_file()
    assert not plt.fignum_exists(fig.number)

    tables = []
    monkeypatch.setattr(
        mlflow, "log_dict", lambda value, path: tables.append((value, path))
    )
    log_polars_table(pl.DataFrame({"score": [1.0]}), "scores.json")
    assert tables == [({"rows": [{"score": 1.0}]}, "scores.json")]

    logged = []
    sentinel = object()

    def log_model(**kwargs):
        logged.append(kwargs)
        return sentinel

    monkeypatch.setattr(mlflow.pyfunc, "log_model", log_model)
    sample = pd.DataFrame({"x": [1.0]})
    assert (
        log_model_from_code(
            source="model.py",
            sample_frame=sample,
            sample_prediction=np.array([2.0]),
            artifacts={"model": "model.bin"},
            metadata={},
            params={"verbose": False},
            name="example",
        )
        is sentinel
    )
    assert logged[0]["input_example"] is sample
    assert logged[0]["params"] == {"verbose": False}
    assert logged[0]["artifacts"] == {"model": "model.bin"}
    for namespace in (
        "ml_toolkit.psi.analysis",
        "ml_toolkit.inflation",
        "ml_toolkit.plots.utils",
        "ml_toolkit.tracking.mlflow",
        "ml_toolkit.tracking.model_artifacts",
    ):
        assert any(
            record.name == namespace and record.levelno == logging.DEBUG
            for record in toolkit_records(caplog)
        )


def test_model_and_shap_debug_logs_preserve_verbosity(caplog):
    caplog.set_level(logging.DEBUG, logger="ml_toolkit")
    params = {
        "iterations": 3,
        "depth": 2,
        "loss_function": "Poisson",
        "verbose": False,
        "allow_writing_files": False,
    }
    model = CatBoostRegressionModel(params).fit(make_data())
    assert params == model.params
    assert model.native_model.get_params()["verbose"] is False
    assert model.predict_pool(make_data()).shape == (24,)
    pytest.importorskip("shap")
    explanation = calculate_shap_values(
        model.native_model, make_data().X, config=ShapConfig(max_samples=4)
    )
    assert explanation.values.shape == (4, 2)
    names = {record.name for record in toolkit_records(caplog)}
    assert "ml_toolkit.models.catboost_regression" in names
    assert "ml_toolkit.explainability" in names
    assert all(record.levelno == logging.DEBUG for record in toolkit_records(caplog))


@pytest.mark.parametrize("classifier", [False, True])
def test_serving_logs_keep_namespace_under_dynamic_loading(
    tmp_path, caplog, monkeypatch, classifier
):
    caplog.set_level(logging.DEBUG, logger="ml_toolkit")
    data = make_data()
    if classifier:
        data = data.with_target(pl.Series("target", [0, 1] * 12))
    factory = CatBoostClassificationModel if classifier else CatBoostRegressionModel
    model = factory(
        {"iterations": 3, "depth": 2, "verbose": False, "allow_writing_files": False}
    ).fit(data)
    path = tmp_path / "model.cbm"
    model.native_model.save_model(str(path))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(model.export_state()))
    registered = []
    monkeypatch.setattr(mlflow.models, "set_model", registered.append)
    source = (
        Path(__file__).resolve().parents[1]
        / "src/ml_toolkit/tracking/entrypoints"
        / f"{factory.__name__}.py"
    )
    runpy.run_path(str(source), run_name="generated_mlflow_module")
    assert len(registered) == 1 and type(registered[0]) is factory
    wrapper = registered[0]
    wrapper.load_context(
        SimpleNamespace(artifacts={"model": str(path), "config": str(config_path)})
    )
    np.testing.assert_allclose(
        wrapper.predict(None, data.to_model_input()), model.predict_pool(data)
    )
    module = "catboost_classification" if classifier else "catboost_regression"
    assert any(
        record.name == f"ml_toolkit.models.{module}"
        and record.getMessage().startswith("Predicting with CatBoost")
        for record in toolkit_records(caplog)
    )


def test_lightgbm_serving_namespace(tmp_path, caplog, monkeypatch):
    caplog.set_level(logging.DEBUG, logger="ml_toolkit")

    class Booster:
        def __init__(self, model_file):
            self.model_file = model_file

        def feature_name(self):
            return ["x"]

        def predict(self, data):
            return data["x"].to_numpy()

    # The optional backend is unnecessary to verify namespace and record delivery.
    monkeypatch.setitem(sys.modules, "lightgbm", SimpleNamespace(Booster=Booster))
    monkeypatch.setattr(mlflow.models, "set_model", lambda model: None)
    source = Path(__file__).resolve().parents[1] / "src/ml_toolkit/models/lightgbm.py"
    loaded = runpy.run_path(
        str(source), run_name="ml_toolkit.models.generated_lightgbm_module"
    )
    assert loaded["logger"].name == "ml_toolkit.models.lightgbm"
    config = tmp_path / "features.json"
    config.write_text(json.dumps({"numerical_features": ["x"], "cat_features": []}))
    wrapper = loaded["LightGBMWrapper"]()
    wrapper.load_context(
        SimpleNamespace(artifacts={"booster": "model.lgb", "features": str(config)})
    )
    np.testing.assert_array_equal(
        wrapper.predict(None, pd.DataFrame({"x": [1.0]})), [1.0]
    )
    assert [record.getMessage() for record in toolkit_records(caplog)] == [
        "Loaded LightGBM serving model",
        "Predicting with LightGBM serving model: 1 rows",
    ]


@pytest.mark.parametrize("level", [logging.DEBUG, logging.WARNING])
@pytest.mark.parametrize("classifier", [False, True])
def test_feature_selection_keeps_catboost_verbosity(caplog, classifier, level):
    caplog.set_level(level, logger="ml_toolkit")
    data = make_data()
    if classifier:
        data = data.with_target(pl.Series("target", [0, 1] * 12))
    fitted = []
    factory = CatBoostClassificationModel if classifier else CatBoostRegressionModel

    def record(params):
        model = factory(params)
        fitted.append(model)
        return model

    params = {"iterations": 3, "verbose": False, "allow_writing_files": False}
    partition = split_holdout(data, HoldoutConfig(val_size=0.25))
    result = RFESelector(
        ranker=importance_ranker, min_features=1, metric="mae", direction="minimize"
    ).select(
        metric=mae,
        train=partition.train,
        validation=partition.val,
        params=params,
        model_factory=record,
    )
    assert [len(step.features) for step in result.steps] == [2, 1]
    assert fitted[0].native_model.get_params()["verbose"] is False
    assert params == fitted[0].params


def test_exceptions_propagate_without_error_records(tmp_path, caplog):
    setup_mlflow(tmp_path)
    caplog.set_level(logging.DEBUG, logger="ml_toolkit")
    error = RuntimeError("model fit failed")

    class FailingModel(MeanModel):
        def fit(self, train, validation=None):
            raise error

    with pytest.raises(RuntimeError) as caught:
        train_model(
            data=make_data(),
            params={},
            model_factory=FailingModel,
            metric=lambda y, p, w: MetricScores({}),
            run_name="failure",
            config=TrainingConfig(validation=TimeSeriesCVConfig(n_splits=2)),
        )
    assert caught.value is error
    assert not any(
        record.levelno >= logging.ERROR for record in toolkit_records(caplog)
    )
    assert not any(
        record.getMessage().startswith("Completed training")
        for record in toolkit_records(caplog)
    )
