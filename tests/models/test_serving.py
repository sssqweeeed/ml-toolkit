"""Portable model artifacts and inference without training dependencies."""

import json
import os
import subprocess
import sys
import sysconfig
import venv
from datetime import date
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml_toolkit.inflation import ConstantRateInflation, ManualInflation
from ml_toolkit.models import (
    CatBoostClassificationModel,
    CatBoostLogLinkRegressionModel,
    CatBoostPUClassificationModel,
    CatBoostRegressionModel,
    InflationAdjustedModel,
)
from ml_toolkit.types import FloatArray
from tests.models.support import PARAMS, SquaredObjective
from tests.support.training import setup_mlflow


@pytest.fixture(scope="session")
def inference_python(tmp_path_factory: pytest.TempPathFactory) -> Path:
    environment = tmp_path_factory.mktemp("inference-environment")
    builder = venv.EnvBuilder(with_pip=False)
    builder.create(environment)
    python = Path(builder.ensure_directories(environment).env_exe)
    library = Path(
        subprocess.check_output(
            [
                str(python),
                "-I",
                "-c",
                "import sysconfig; print(sysconfig.get_path('purelib'))",
            ],
            text=True,
        ).strip()
    )
    # Reuse installed runtime libraries without installing the toolkit or training
    # dependencies. Exclude .pth files so editable installs cannot expose source.
    for entry in Path(sysconfig.get_path("purelib")).iterdir():
        if entry.suffix == ".pth" or entry.name.startswith(
            ("optuna", "numba", "llvmlite", "ml_toolkit", "__editable__")
        ):
            continue
        (library / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
    return python


def check_bundled_inference(
    python: Path,
    tmp_path: Path,
    directory: Path,
    frame: pd.DataFrame,
    expected: FloatArray,
    kind: str,
    nested: str = "",
) -> None:
    requirements = (directory / "requirements.txt").read_text().lower()
    assert not any(
        line.startswith(("optuna", "numba", "ml-toolkit", "ml_toolkit"))
        for line in requirements.splitlines()
    )
    frame.to_pickle(tmp_path / "frame.pkl")
    np.save(tmp_path / "prediction.npy", expected)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            str(python),
            "-I",
            str(Path(__file__).resolve().parents[1] / "check_model_inference.py"),
            str(tmp_path),
            mlflow.get_tracking_uri(),
            str(directory),
            kind,
            nested,
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "kind",
    [
        "regression",
        "custom",
        "log_link",
        "gamma_native",
        "gamma_log_link",
        "gamma_raw",
        "classification",
        "classification_auto",
        "uPU",
        "nnPU",
        "inflation_native",
        "inflation_log_link",
        "inflation_gamma",
    ],
)
def test_models_from_code_roundtrip_outside_repository(
    data, tmp_path, kind, inference_python
):
    setup_mlflow(tmp_path)
    if kind in ("classification", "classification_auto", "uPU", "nnPU"):
        data = data.with_target(pl.Series("target", [1, 0, 0] * 16))
    if kind == "regression":
        model = CatBoostRegressionModel(PARAMS, prediction_type="RawFormulaVal")
    elif kind == "custom":
        model = CatBoostRegressionModel(
            {
                **PARAMS,
                "loss": SquaredObjective(),
                "eval_metric": "RMSE",
                "leaf_estimation_method": "Gradient",
            },
        )
    elif kind == "log_link":
        model = CatBoostLogLinkRegressionModel(PARAMS)
    elif kind.startswith("gamma_"):
        factory = (
            CatBoostRegressionModel
            if kind == "gamma_native"
            else CatBoostLogLinkRegressionModel
        )
        model = factory(
            {**PARAMS, "loss_function": "Gamma"},
            prediction_type="RawFormulaVal" if kind == "gamma_raw" else None,
        )
    elif kind == "classification":
        model = CatBoostClassificationModel({**PARAMS, "decision_threshold": 0.4})
    elif kind == "classification_auto":
        model = CatBoostClassificationModel(PARAMS)
    elif kind in ("uPU", "nnPU"):
        model = CatBoostPUClassificationModel(
            {**PARAMS, "loss": kind, "pu_prior": 0.2, "decision_threshold": 0.3},
        )
    else:
        regressor_factory = (
            CatBoostRegressionModel
            if kind == "inflation_native"
            else CatBoostLogLinkRegressionModel
        )
        regressor = regressor_factory(
            {**PARAMS, "loss_function": "Gamma"} if kind == "inflation_gamma" else PARAMS
        )
        inflation = (
            ConstantRateInflation(0.03)
            if kind == "inflation_native"
            else ManualInflation({date(2024, 1, 1): 1.0, date(2025, 1, 1): 1.1})
        )
        model = InflationAdjustedModel(regressor, inflation, date(2024, 1, 1))
    model.fit(data.take(range(32)), data.take(range(32, 48)))
    frame = data.to_model_input()
    # Nonstandard index and shuffled feature order must not disturb dates or categories.
    frame.index = pd.Index(np.arange(len(frame)) + 100)
    frame = frame.loc[:, list(reversed(frame.columns))]
    if isinstance(model, InflationAdjustedModel):
        frame[model.prediction_date_column] = pd.Series(
            data.time.to_list(), index=frame.index, dtype="datetime64[ns]"
        )
    expected = model.predict_pool(data)
    np.testing.assert_allclose(model.predict_frame(frame), expected)
    with mlflow.start_run():
        info = model.log(data)
    if isinstance(info, tuple):
        info = info[1]
    directory = Path(mlflow.artifacts.download_artifacts(info.model_uri))
    files = list((directory / "code").rglob("*"))
    assert not any(
        path.name in ("__pycache__", ".codex", ".codegraph", "AGENTS.md")
        for path in files
    )
    if kind == "classification_auto":
        loaded = mlflow.pyfunc.load_model(info.model_uri).unwrap_python_model()
        assert loaded.threshold == model.threshold
        assert loaded.validation_f1 == model.validation_f1
        assert model.validation_f1 is not None
        from sklearn.metrics import f1_score

        validation = data.take(range(32, 48))
        assert loaded.validation_f1 == pytest.approx(
            f1_score(
                validation.y.to_numpy(),
                model.predict_pool(validation) >= model.threshold,
                sample_weight=validation.weight.to_numpy(),
            )
        )
    state = model.export_state()
    assert state["version"] == 1 and state["model_type"] == type(model).__name__
    nested = (
        type(model.model).__name__ if isinstance(model, InflationAdjustedModel) else ""
    )
    check_bundled_inference(
        inference_python,
        tmp_path,
        directory,
        frame,
        expected,
        type(model).__name__,
        nested,
    )


def test_importing_model_modules_does_not_register_models():
    code = """
import mlflow
calls = []
mlflow.models.set_model = calls.append
import ml_toolkit.models
assert calls == []
"""
    result = subprocess.run(
        [sys.executable, "-c", code], text=True, capture_output=True, timeout=60
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "update",
    [
        {"version": 2},
        {"model_type": "CatBoostRegressionModel"},
        {"baseline_mode": "raw_prediction"},
    ],
    ids=["version", "model-type", "baseline-policy"],
)
def test_loading_rejects_wrong_state_version_or_policy(data, tmp_path, update):
    from types import SimpleNamespace

    model = CatBoostLogLinkRegressionModel(PARAMS).fit(data)
    path = tmp_path / "model.cbm"
    model.native_model.save_model(str(path))
    state = model.export_state()
    config = tmp_path / "config.json"
    config.write_text(json.dumps({**state, **update}))
    with pytest.raises(ValueError, match="state|baseline"):
        CatBoostLogLinkRegressionModel().load_context(
            SimpleNamespace(artifacts={"model": str(path), "config": str(config)})
        )


def test_optional_lightgbm_models_from_code(data, tmp_path, inference_python):
    lightgbm = pytest.importorskip("lightgbm")
    from ml_toolkit.models.lightgbm import LightGBMServingModel

    setup_mlflow(tmp_path)
    frame = data.X.select("noise", "signal").to_pandas()
    estimator = lightgbm.LGBMRegressor(n_estimators=3, verbosity=-1, n_jobs=1).fit(
        frame, data.y.to_numpy()
    )
    model = LightGBMServingModel(estimator, numerical_features=list(frame.columns))
    with mlflow.start_run():
        info = model.log(data)
    loaded = mlflow.pyfunc.load_model(info.model_uri)
    assert type(loaded.unwrap_python_model()).__name__ == "LightGBMServingModel"
    np.testing.assert_allclose(loaded.predict(frame), model.predict(None, frame))
    directory = Path(mlflow.artifacts.download_artifacts(info.model_uri))
    check_bundled_inference(
        inference_python,
        tmp_path,
        directory,
        frame,
        model.predict(None, frame),
        "LightGBMServingModel",
    )
