"""Adapter feature selection and configured model packaging."""

from datetime import date

import mlflow
import polars as pl
import pytest
from tests.support.training import mae, make_data

from ml_toolkit.feature_selection import RFESelector
from ml_toolkit.config import HoldoutConfig, TrackingConfig
from ml_toolkit.inflation import ConstantRateInflation
from ml_toolkit.models import CatBoostRegressionModel, InflationAdjustedModel
from ml_toolkit.training import importance_ranker, split_holdout


@pytest.mark.parametrize("classifier", [False, True])
def test_selection_fits_native_model_with_normal_parameters(classifier):
    from ml_toolkit.models import CatBoostClassificationModel

    data = make_data()
    if classifier:
        data = data.with_target(pl.Series("target", [0, 1] * 12))
    fitted = []
    factory = CatBoostClassificationModel if classifier else CatBoostRegressionModel

    def record(params):
        model = factory(params)
        fitted.append(model)
        return model

    params = {"thread_count": 1, "iterations": 3, "verbose": False, "allow_writing_files": False}
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
    assert result.features == result.steps[result.selected_step].features
    assert fitted[0].native_model.tree_count_ > 0
    assert fitted[0].params == params
    assert not hasattr(fitted[0], "select_features")


def test_standalone_model_logging_uses_example_size(monkeypatch):
    data = make_data()
    model = CatBoostRegressionModel(
        {"thread_count": 1, "iterations": 2, "depth": 2, "verbose": False, "allow_writing_files": False}
    ).fit(data)
    examples = []
    monkeypatch.setattr(
        model,
        "log_artifacts",
        lambda frame, predictions, **kwargs: examples.append((frame, kwargs)),
    )
    monkeypatch.setattr(mlflow, "log_param", lambda *a, **k: None)
    monkeypatch.setattr(mlflow, "log_dict", lambda *a, **k: None)
    model.log(data, config=TrackingConfig(input_example_rows=2, model_name="two-rows"))
    assert len(examples[0][0]) == 2 and examples[0][1]["name"] == "two-rows"


def test_inflation_wrapper_uses_shared_importance_selection():

    data = make_data()
    fitted = []

    def factory(params):
        model = InflationAdjustedModel(
            CatBoostRegressionModel(params),
            ConstantRateInflation(0.03),
            date(2024, 1, 1),
        )
        fitted.append(model)
        return model

    partition = split_holdout(data, HoldoutConfig(val_size=0.25))
    result = RFESelector(
        ranker=importance_ranker, min_features=1, metric="mae", direction="minimize"
    ).select(
        metric=mae,
        train=partition.train,
        validation=partition.val,
        params={"thread_count": 1, "iterations": 3, "verbose": False, "allow_writing_files": False},
        model_factory=factory,
    )
    assert [len(step.features) for step in result.steps] == [2, 1]
    assert result.features == result.steps[result.selected_step].features
    assert fitted[0].native_model.tree_count_ > 0
    assert not hasattr(fitted[0], "select_features")
