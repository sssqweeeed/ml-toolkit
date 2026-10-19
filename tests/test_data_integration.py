from datetime import date

import mlflow
import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from test_training import MeanModel, ProjectData, mae, make_data, setup_mlflow

from ml_toolkit.feature_selection import RFESelector
from ml_toolkit.config import (
    PostFitEvaluationConfig,
    TimeSeriesCVConfig,
    TrainingConfig,
)
from examples.project_pool import ProjectPool as Pool
from ml_toolkit.inflation import ConstantRateInflation
from ml_toolkit.metrics import metrics_classification
from ml_toolkit.models import (
    CatBoostClassificationModel,
    CatBoostRegressionModel,
    InflationAdjustedModel,
)
from ml_toolkit.training import importance_ranker, make_validation_splits, train_model


@pytest.fixture
def pool():
    data = make_data()
    return Pool(
        X=data.X.with_columns(pl.Series("category", [10, None, 20] * 8)),
        y=data.y,
        time=data.time,
        weight=data.weight,
        business_type=pl.Series("business_type", ["a", "b"] * 12),
        cat_features=("category",),
        numerical_features=("x", "unused"),
    )


@pytest.mark.parametrize("classifier", [False, True])
def test_catboost_predictions_use_same_categories_as_fitting(pool, classifier):
    original = pool.X.clone()
    if classifier:
        pool = pool.with_target(pl.Series("target", [0, 1] * 12))
    factory = CatBoostClassificationModel if classifier else CatBoostRegressionModel
    model = factory(
        {
            "iterations": 3,
            "depth": 2,
            "verbose": False,
            "thread_count": 1,
            "allow_writing_files": False,
        }
    )
    model.fit(pool.take(range(16)), pool.take(range(16, 24)))
    prediction = model.predict_pool(pool)
    assert prediction.shape == (24,)
    assert np.isfinite(prediction).all()
    np.testing.assert_allclose(prediction, model.predict_frame(pool.to_model_input()))
    if classifier:
        np.testing.assert_array_equal(
            model.predict_labels_pool(pool), model.predict_labels(pool.to_model_input())
        )
    assert_frame_equal(pool.X, original)


def test_training_feature_selection_accepts_project_pool(pool, tmp_path):
    setup_mlflow(tmp_path)
    fitted = []

    class RecordingModel(MeanModel):
        def fit(self, train, validation=None):
            assert isinstance(train, Pool)
            if len(train.X.columns) == 1:
                assert train.numerical_features == ("x",)
                assert train.cat_features == ()
            assert len(train.business_type) == len(train)
            fitted.append(train)
            return super().fit(train, validation)

    result = train_model(
        data=pool,
        params={},
        model_factory=RecordingModel,
        metric=mae,
        run_name="shared-pool-selection",
        config=TrainingConfig(validation=TimeSeriesCVConfig(n_splits=2)),
        feature_selector=RFESelector(
            ranker=importance_ranker, min_features=1, metric="mae", direction="minimize"
        ),
    )
    assert result.features == ("x",)
    assert len(result.validation.splits) == 2
    assert len(fitted[-1]) == len(pool)


def test_independent_project_container_preserves_its_type_through_model_fitting():
    data = make_data()
    partitions = make_validation_splits(data, TimeSeriesCVConfig(n_splits=2))
    for part in partitions:
        assert isinstance(part.train, ProjectData)
        assert isinstance(part.val, ProjectData)
        model = CatBoostRegressionModel(
            {
                "iterations": 3,
                "depth": 2,
                "thread_count": 1,
                "verbose": False,
                "allow_writing_files": False,
            }
        ).fit(part.train, part.val)
        predictions = model.predict_pool(part.val)
        assert predictions.shape == (len(part.val),)
        assert np.isfinite(predictions).all()
        np.testing.assert_allclose(
            predictions, model.predict_frame(part.val.to_model_input())
        )


@pytest.mark.parametrize("classifier", [False, True])
def test_catboost_pool_training_and_serving_roundtrip(pool, tmp_path, classifier):
    setup_mlflow(tmp_path)
    if classifier:
        pool = pool.with_target(pl.Series("target", [0, 1] * 12))
    result = train_model(
        data=pool,
        params={
            "iterations": 3,
            "depth": 2,
            "verbose": False,
            "thread_count": 1,
            "allow_writing_files": False,
        },
        model_factory=(
            CatBoostClassificationModel if classifier else CatBoostRegressionModel
        ),
        metric=metrics_classification if classifier else mae,
        run_name="shared-pool-catboost",
        config=TrainingConfig(validation=TimeSeriesCVConfig(n_splits=2)),
    )
    runs = mlflow.search_runs(output_format="list")
    final = next(
        run for run in runs if run.data.tags.get("mlflow.runName") == "final_fit"
    )
    artifact = "model"
    loaded = mlflow.pyfunc.load_model(f"runs:/{final.info.run_id}/{artifact}")
    np.testing.assert_allclose(
        loaded.predict(pool.to_model_input()),
        result.model.predict_pool(pool),
    )


def test_inflation_logging_keeps_feature_rows_aligned_with_dates(
    pool, tmp_path, monkeypatch
):
    setup_mlflow(tmp_path)
    regression = CatBoostRegressionModel(
        {
            "iterations": 3,
            "depth": 2,
            "verbose": False,
            "thread_count": 1,
            "allow_writing_files": False,
        }
    )
    model = InflationAdjustedModel(
        regression, ConstantRateInflation(0.03), date(2024, 1, 1)
    )
    model.fit(pool)
    examples = []
    import ml_toolkit.tracking._model_state as module

    log_artifacts = module.log_catboost

    def capture(adapter, frame, prediction, **kwargs):
        examples.append((frame.copy(), prediction.copy()))
        return log_artifacts(adapter, frame, prediction, **kwargs)

    monkeypatch.setattr(module, "log_catboost", capture)
    with mlflow.start_run():
        _, info = model.log(pool)
    frame, prediction = examples[-1]
    assert frame["x"].to_list() == pool.X["x"].head(5).to_list()
    assert (
        frame[model.prediction_date_column].dt.date.to_list()
        == pool.time.head(5).to_list()
    )
    np.testing.assert_allclose(prediction, model.predict_pool(pool.take(range(5))))
    loaded = mlflow.pyfunc.load_model(info.model_uri)
    np.testing.assert_allclose(loaded.predict(frame), prediction)
