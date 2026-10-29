"""Adapter prefixes preserve transformations, independence and portable state."""

from datetime import date
import json
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from ml_toolkit.inflation import (
    ConstantRateInflation,
    LogLinearInflation,
    SplineInflation,
)
from ml_toolkit.models import (
    CatBoostClassificationModel,
    CatBoostLogLinkRegressionModel,
    CatBoostPUClassificationModel,
    CatBoostRegressionModel,
    InflationAdjustedModel,
    ShrinkableModel,
)
from tests.models.support import PARAMS


def test_curve_uses_early_stopped_fitted_size(data):
    from ml_toolkit.model_evaluation import ShrinkCurve
    from tests.support.training import mae

    train = data.take(range(32))
    validation = data.take(range(32, 48)).with_target(pl.Series([-10.0] * 16))
    model = CatBoostRegressionModel(
        {**PARAMS, "iterations": 50, "early_stopping_rounds": 3}
    ).fit(train, validation)
    assert 1 <= model.n_iterations < 50
    table = ShrinkCurve(step=10).calculate(
        model=model,
        metric=mae,
        data=validation,
        split=pl.Series(["val"] * len(validation)),
    )
    assert table["n_iterations"].max() == model.native_model.tree_count_
    with pytest.raises(ValueError, match="exceed the fitted model"):
        ShrinkCurve(sizes=(model.n_iterations + 1,)).calculate(
            model=model,
            metric=mae,
            data=validation,
            split=pl.Series(["val"] * len(validation)),
        )


def test_shrinking_keeps_automatically_selected_threshold(data):
    data = data.with_target(pl.Series("target", [0, 1] * 24))
    model = CatBoostClassificationModel({**PARAMS, "use_best_model": False}).fit(
        data, data
    )
    threshold, validation_f1 = model.threshold, model.validation_f1
    assert validation_f1 is not None
    small = model.shrink(2)
    assert small.threshold == threshold and small.validation_f1 is None
    assert model.threshold == threshold and model.validation_f1 == validation_f1
    assert small.get_refit_params()["decision_threshold"] == threshold


@pytest.mark.parametrize("factory", [LogLinearInflation, SplineInflation])
def test_shrunk_inflation_copies_learned_state(data, tmp_path, factory):
    model = InflationAdjustedModel(
        CatBoostRegressionModel(PARAMS), factory(control_features=[]), date(2024, 1, 1)
    ).fit(data)
    before = model.inflation.to_config()
    expected = model.predict_pool(data)
    small = model.shrink(2)
    assert small.inflation.to_config() == before
    path = tmp_path / "model.cbm"
    small.native_model.save_model(str(path))
    config = tmp_path / "config.json"
    config.write_text(json.dumps(small.export_state()))
    loaded = InflationAdjustedModel()
    loaded.load_context(
        SimpleNamespace(artifacts={"model": str(path), "config": str(config)})
    )
    np.testing.assert_array_equal(loaded.predict_pool(data), small.predict_pool(data))
    small.inflation.fit(data.with_target(pl.Series([2.0] * len(data))))
    assert model.inflation.to_config() == before
    np.testing.assert_array_equal(model.predict_pool(data), expected)


@pytest.mark.parametrize(
    "kind",
    [
        "regression",
        "log_link",
        "gamma",
        "gamma_raw",
        "gamma_log_link",
        "classification",
        "uPU",
        "nnPU",
        "inflation",
        "inflation_log_link",
    ],
)
def test_fitted_prefix_matches_native_predictions_and_roundtrips(data, tmp_path, kind):
    if kind in ("classification", "uPU", "nnPU"):
        data = data.with_target(pl.Series("target", [1, 0, 0] * 16))
        factory = (
            CatBoostClassificationModel
            if kind == "classification"
            else CatBoostPUClassificationModel
        )
        extras = {} if kind == "classification" else {"loss": kind, "pu_prior": 0.2}
        model = factory({**PARAMS, **extras, "decision_threshold": 0.4})
    else:
        factory = (
            CatBoostLogLinkRegressionModel
            if kind in ("log_link", "gamma_log_link", "inflation_log_link")
            else CatBoostRegressionModel
        )
        params = {
            **PARAMS,
            **({"loss_function": "Gamma"} if kind.startswith("gamma") else {}),
        }
        model = factory(
            params, prediction_type="RawFormulaVal" if kind == "gamma_raw" else None
        )
        if kind.startswith("inflation"):
            model = InflationAdjustedModel(
                model, ConstantRateInflation(0.03), date(2024, 1, 1)
            )
    model.fit(data, data)
    assert isinstance(model, ShrinkableModel)
    original = model.predict_pool(data).copy()
    state = json.loads(json.dumps(model.export_state()))
    small = model.shrink(2)
    assert type(small) is type(model)
    assert small.n_iterations == 2 and model.n_iterations == 5
    assert small.native_model is not model.native_model
    assert small.get_refit_params()["iterations"] == 2
    if model.best_iteration is not None and model.best_iteration >= 2:
        assert small.best_iteration is None
    base = model.model if kind.startswith("inflation") else model
    frame = data.to_model_input()
    if kind in ("classification", "uPU", "nnPU"):
        raw = (
            model.native_model.predict(
                frame, ntree_end=2, prediction_type="RawFormulaVal"
            )
            + model.baseline
        )
        expected = np.exp(-np.logaddexp(0.0, -raw))
        assert small.threshold == model.threshold
        assert small.validation_f1 is None
    else:
        options = (
            {}
            if base._effective_prediction_type is None
            else {"prediction_type": base._effective_prediction_type}
        )
        expected = model.native_model.predict(frame, ntree_end=2, **options)
        if kind.startswith("inflation"):
            expected = model.inflation.adjust(
                expected,
                pl.Series([model.reference_date] * len(data)),
                data.time,
            )
            assert small.inflation is not model.inflation
            assert small.inflation.to_config() == model.inflation.to_config()
    np.testing.assert_allclose(small.predict_pool(data), expected, rtol=1e-12)
    np.testing.assert_array_equal(original, model.predict_pool(data))
    assert json.loads(json.dumps(model.export_state())) == state
    full = model.shrink(model.n_iterations)
    np.testing.assert_array_equal(full.predict_pool(data), original)
    assert full.native_model is not model.native_model
    # Restore exactly the state and native files used by MLflow packaging.
    model_file = tmp_path / "model.cbm"
    small.native_model.save_model(str(model_file))
    config = tmp_path / "state.json"
    config.write_text(json.dumps(small.export_state()))
    loaded = type(small)()
    loaded.load_context(
        SimpleNamespace(artifacts={"model": str(model_file), "config": str(config)})
    )
    assert loaded.n_iterations == 2
    assert loaded.best_iteration == small.best_iteration
    assert loaded.get_refit_params()["iterations"] == 2
    np.testing.assert_array_equal(loaded.predict_pool(data), small.predict_pool(data))
    assert loaded.shrink(1).n_iterations == 1
    # Refit resets retained-size metadata, using constructor settings again.
    small.fit(data)
    assert small.n_iterations == 5
    assert small.get_refit_params()["iterations"] == 5


@pytest.mark.parametrize(
    "factory", [CatBoostRegressionModel, CatBoostClassificationModel]
)
def test_invalid_sizes_and_unfitted_adapters(data, factory):
    model = factory(PARAMS)
    with pytest.raises(RuntimeError, match="fitted"):
        model.shrink(1)
    if factory is CatBoostClassificationModel:
        data = data.with_target(pl.Series("target", [0, 1] * 24))
    model.fit(data)
    for size in (0, -1, 6, 2.0, True, None):
        with pytest.raises(ValueError, match="integer between"):
            model.shrink(size)


def test_preserved_best_iteration_and_independent_constructor_settings(data):
    model = CatBoostClassificationModel(PARAMS).fit(
        data.with_target(pl.Series("target", [0, 1] * 24))
    )
    model._best_iteration = 1
    small = model.shrink(3)
    assert small.best_iteration == 1
    assert small.get_refit_params()["iterations"] == 3
    small.params["iterations"] = 99
    assert model.params["iterations"] == 5


@pytest.mark.parametrize("estimator", [False, True])
def test_lightgbm_prefix_respects_serving_and_best_iteration(data, tmp_path, estimator):
    lgb = pytest.importorskip("lightgbm")
    from ml_toolkit.models.lightgbm import LightGBMServingModel

    data = data.select_features(("noise", "signal"))
    frame = data.to_model_input()
    if estimator:
        native = lgb.LGBMRegressor(
            n_estimators=5, num_threads=1, verbosity=-1, min_child_samples=2
        ).fit(frame, data.y.to_numpy())
        booster = native.booster_
    else:
        native = lgb.train(
            {
                "objective": "regression",
                "num_threads": 1,
                "verbosity": -1,
                "min_data_in_leaf": 2,
            },
            lgb.Dataset(frame, label=data.y.to_numpy()),
            num_boost_round=5,
        )
        booster = native
    booster.best_iteration = 3
    model = LightGBMServingModel(native, numerical_features=list(frame.columns))
    before = model.predict_pool(data)
    assert model.n_iterations == 3
    assert isinstance(model, ShrinkableModel)
    small = model.shrink(2)
    assert small.n_iterations == 2
    np.testing.assert_allclose(
        small.predict_pool(data), booster.predict(frame, num_iteration=2)
    )
    np.testing.assert_array_equal(before, model.predict_pool(data))
    assert model.native_model.current_iteration() == 5
    small.numerical_features.append("new")
    assert "new" not in model.numerical_features
    with pytest.raises(ValueError, match="between 1 and 3"):
        model.shrink(4)
    np.testing.assert_allclose(model.shrink(3).predict_pool(data), before)
    path = tmp_path / "model.txt"
    small.native_model.save_model(str(path))
    restored = LightGBMServingModel(lgb.Booster(model_file=str(path)))
    assert restored.n_iterations == 2
    np.testing.assert_allclose(restored.predict_pool(data), small.predict_pool(data))


def test_lightgbm_shrink_preserves_categorical_preparation(data):
    lgb = pytest.importorskip("lightgbm")
    from ml_toolkit.models.lightgbm import LightGBMServingModel

    frame = data.to_model_input()
    frame["category"] = frame["category"].astype("category")
    booster = lgb.train(
        {
            "objective": "regression",
            "num_threads": 1,
            "verbosity": -1,
            "min_data_in_leaf": 2,
        },
        lgb.Dataset(frame, label=data.y.to_numpy()),
        num_boost_round=5,
    )
    model = LightGBMServingModel(
        booster, numerical_features=["noise", "signal"], cat_features=["category"]
    )
    np.testing.assert_allclose(
        model.shrink(2).predict_pool(data), booster.predict(frame, num_iteration=2)
    )
