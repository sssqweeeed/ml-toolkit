"""The complete example adapter preserves its transformation during serving."""

from datetime import date, timedelta

import mlflow
import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_series_equal

from examples.custom_model import TargetScaledRegressor, model_factory, weighted_mae
from examples.project_pool import ProjectPool
from ml_toolkit import train_model
from ml_toolkit.config import HoldoutConfig, TrackingConfig, TrainingConfig
from ml_toolkit.models import CatBoostRegressionModel
from ml_toolkit.training import split_holdout


@pytest.fixture
def data():
    count = 36
    values = np.linspace(0.0, 1.0, count)
    return ProjectPool(
        X=pl.DataFrame(
            {
                "x": values,
                "region": ["north", None, "south"] * (count // 3),
                "other": np.sin(np.arange(count)),
            }
        ),
        y=pl.Series("target", 1500.0 + 300.0 * values),
        time=pl.Series(
            "date", [date(2025, 1, 1) + timedelta(days=i) for i in range(count)]
        ),
        weight=pl.Series("weight", np.linspace(0.5, 2.0, count)),
        cat_features=("region",),
        numerical_features=("x", "other"),
        object_id=pl.Series("id", np.arange(count)),
    )


@pytest.fixture
def params():
    return {
        "iterations": 12,
        "depth": 2,
        "loss_function": "RMSE",
        "early_stopping_rounds": 3,
        "random_seed": 42,
        "thread_count": 1,
        "verbose": False,
        "allow_writing_files": False,
    }


@pytest.fixture
def tracking_store(tmp_path):
    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'tracking.db'}")
    experiment = mlflow.create_experiment(
        "custom-model-test", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    mlflow.set_experiment(experiment_id=experiment)
    yield
    mlflow.set_tracking_uri(previous_uri)


def test_adapter_scales_both_fit_samples_and_inverts_predictions(data, params):
    original_target = data.y.clone()
    part = split_holdout(data, HoldoutConfig(val_size=0.25))
    scale = 250.0
    adapter = TargetScaledRegressor(params, target_scale=scale).fit(part.train, part.val)
    native_reference = CatBoostRegressionModel(params).fit(
        part.train.with_target(part.train.y / scale),
        part.val.with_target(part.val.y / scale),
    )
    native_predictions = native_reference.predict_pool(data)
    np.testing.assert_allclose(
        adapter.native_model.predict(data.to_model_input()), native_predictions
    )
    np.testing.assert_allclose(adapter.predict_pool(data), native_predictions * scale)
    assert adapter.best_iteration == native_reference.best_iteration
    importances = adapter.feature_importances(data)
    assert list(importances) == data.X.columns
    assert np.isfinite(list(importances.values())).all()
    assert_series_equal(data.y, original_target)


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_adapter_rejects_invalid_target_scale(params, scale):
    with pytest.raises(ValueError, match="finite and positive"):
        TargetScaledRegressor(params, target_scale=scale)


def test_workflow_refits_a_fresh_adapter_with_validation_iteration_budget(
    data, params, tracking_store
):
    original_params = params.copy()
    adapters = []

    def factory(settings):
        adapter = model_factory(settings)
        adapters.append(adapter)
        return adapter

    result = train_model(
        data=data,
        params=params,
        model_factory=factory,
        metric=weighted_mae,
        run_name="custom-model-refit",
        config=TrainingConfig(
            validation=HoldoutConfig(val_size=0.25),
            test_size=0.2,
            tracking=TrackingConfig(log_final_model=False),
        ),
    )
    assert len(adapters) == 2
    validation_adapter, final_adapter = adapters
    assert result.model is final_adapter
    assert validation_adapter is not final_adapter
    assert validation_adapter.native_model is not final_adapter.native_model
    assert validation_adapter.target_scale == final_adapter.target_scale == 1000.0
    assert validation_adapter.best_iteration is not None
    assert result.params["iterations"] == validation_adapter.best_iteration + 1
    assert final_adapter.native_model.tree_count_ == result.params["iterations"]
    assert result.params["use_best_model"] is False
    assert "early_stopping_rounds" not in result.params
    assert params == original_params
    assert result.test_metrics is not None
    assert np.isfinite(result.test_metrics["mae"])


def test_logged_adapter_restores_scale_categories_and_feature_order(
    data, params, tracking_store
):
    adapter = model_factory(params).fit(data)
    with mlflow.start_run(run_name="custom-model-serving"):
        info = adapter.log(data, config=TrackingConfig(input_example_rows=3))
    served = mlflow.pyfunc.load_model(info.model_uri)
    reordered = data.X.select("region", "other", "x").to_pandas()
    reordered["extra"] = -1.0
    original_frame = reordered.copy(deep=True)
    np.testing.assert_allclose(served.predict(reordered), adapter.predict_pool(data))
    pd.testing.assert_frame_equal(reordered, original_frame)
