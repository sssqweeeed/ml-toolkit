"""PSI references travel with each model, including registry and local copies."""

from datetime import date
from tempfile import TemporaryDirectory

import mlflow
import numpy as np
import pytest
from polars.testing import assert_frame_equal

from examples.custom_model import TargetScaledRegressor
from ml_toolkit.config import TrackingConfig
from ml_toolkit.feature_levels import FeatureLevelConfig
from ml_toolkit.inflation import ConstantRateInflation
from ml_toolkit.models import (
    CatBoostClassificationModel,
    CatBoostRegressionModel,
    InflationAdjustedModel,
)
from ml_toolkit.psi import PSIBaseline, PSIConfig
from ml_toolkit.tracking import load_psi_baseline
from ml_toolkit.tracking.model_artifacts import merge_artifacts
from tests.support.training import make_data


@pytest.fixture
def store(tmp_path):
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'tracking.db'}")
    mlflow.set_experiment(
        experiment_id=mlflow.create_experiment(
            "psi-models", artifact_location=(tmp_path / "artifacts").as_uri()
        )
    )
    yield
    mlflow.set_tracking_uri(previous)


@pytest.mark.parametrize(
    "kind", ["regression", "classification", "inflation", "custom", "lightgbm"]
)
def test_model_artifact_roundtrip_and_registry(kind, store, tmp_path, monkeypatch):
    data = make_data()
    params = {
        "iterations": 2,
        "depth": 2,
        "verbose": False,
        "thread_count": 1,
        "allow_writing_files": False,
    }
    if kind == "regression":
        model = CatBoostRegressionModel(params).fit(data)
    elif kind == "classification":
        data = data.with_target((data.y > data.y.mean()).cast(float))
        model = CatBoostClassificationModel(params).fit(data)
    elif kind == "inflation":
        model = InflationAdjustedModel(
            CatBoostRegressionModel(params),
            ConstantRateInflation(0.03),
            date(2024, 1, 1),
        ).fit(data)
    elif kind == "custom":
        model = TargetScaledRegressor(params).fit(data)
    else:
        lgb = pytest.importorskip("lightgbm")
        from ml_toolkit.models.lightgbm import LightGBMServingModel

        booster = lgb.train(
            {"verbosity": -1, "num_threads": 1, "min_data_in_leaf": 2},
            lgb.Dataset(data.X.to_pandas(), label=data.y.to_numpy()),
            num_boost_round=2,
        )
        model = LightGBMServingModel(
            booster, numerical_features=tuple(data.X.columns), cat_features=()
        )
    baseline = PSIBaseline.from_reference(
        data.X,
        config=PSIConfig(
            tuple(FeatureLevelConfig(n, "numerical") for n in data.X.columns)
        ),
        time=data.time,
    )
    with TemporaryDirectory() as directory, mlflow.start_run(run_name=kind) as run:
        baseline.save(directory)
        infos = model.log(
            data,
            config=TrackingConfig(registered_model_name=f"psi-{kind}"),
            extra_artifacts={"psi_baseline": directory},
        )
        run_id = run.info.run_id
    infos = infos if isinstance(infos, tuple) else (infos,)
    info = infos[-1]
    local = mlflow.artifacts.download_artifacts(
        artifact_uri=info.model_uri, dst_path=str(tmp_path / "download")
    )
    frame = data.to_model_input(categorical_as_strings=True)
    if kind == "inflation":
        frame[model.prediction_date_column] = data.time.to_pandas()
    served = mlflow.pyfunc.load_model(info.model_uri)
    np.testing.assert_allclose(
        served.predict(frame),
        model.predict(None, frame) if kind == "lightgbm" else model.predict_pool(data),
    )
    monkeypatch.setattr(
        mlflow.pyfunc,
        "load_model",
        lambda *a, **k: pytest.fail("must not load predictor"),
    )
    for uri in [
        *(i.model_uri for i in infos),
        f"models:/psi-{kind}/1",
        local,
        f"runs:/{run_id}/model",
    ]:
        loaded = load_psi_baseline(uri)
        assert_frame_equal(loaded.reference_counts, baseline.reference_counts)
        assert_frame_equal(loaded.compare(data.X).psi, baseline.compare(data.X).psi)


def test_missing_baseline_and_conflict(store):
    data = make_data()
    model = CatBoostRegressionModel(
        {
            "iterations": 2,
            "verbose": False,
            "thread_count": 1,
            "allow_writing_files": False,
        }
    ).fit(data)
    with mlflow.start_run(run_name="no-baseline"):
        info = model.log(data)
    with pytest.raises(FileNotFoundError, match="no PSI baseline"):
        load_psi_baseline(info.model_uri)
    assert mlflow.pyfunc.load_model(info.model_uri).predict(
        data.X.to_pandas()
    ).shape == (len(data),)
    with pytest.raises(ValueError, match="conflict"):
        merge_artifacts({"model": "weights.bin"}, {"model": "other.bin"})


def test_invalid_artifact_path(tmp_path):
    (tmp_path / "MLmodel").write_text(
        "flavors:\n  python_function:\n    artifacts:\n      psi_baseline:\n        path: ../outside\n"
    )
    with pytest.raises(ValueError, match="relative"):
        load_psi_baseline(str(tmp_path))
