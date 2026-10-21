"""Gamma mathematics, training policies, persistence and CUDA contracts."""

import json
import math
import os
import subprocess
import sys
import warnings
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import catboost
import numpy as np
import polars as pl
import pytest
from sklearn.metrics import mean_gamma_deviance

from ml_toolkit.models import CatBoostLogLinkRegressionModel, CatBoostRegressionModel
from ml_toolkit.models.losses.gamma.cpu import (
    GammaMetric,
    GammaObjective,
    gamma_derivatives,
    gamma_deviance,
)
from tests.models.support import PARAMS


class ReferenceGammaObjective:
    no_jit = True

    def calc_ders_range(self, approxes, targets, weights):
        result = []
        for i, (a, y) in enumerate(zip(approxes, targets)):
            w = 1.0 if weights is None else float(weights[i])
            ratio = float(y) * math.exp(-float(a))
            result.append((w * (ratio - 1), -w * ratio))
        return result


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("weighted", [False, True])
def test_gamma_derivatives_and_deviance(dtype, weighted):
    y = np.array([1e-30, 0.1, 1, 10, 1e30], dtype=dtype)
    a = np.log(y.astype(float)) + np.array([-2, -0.1, 0, 0.2, 2])
    w = np.array([0.5, 0, 3, 1, 2], dtype=dtype) if weighted else None
    actual = gamma_derivatives(a, y, w)
    expected = ReferenceGammaObjective().calc_ders_range(a, y, w)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
    eps = 1e-4
    weights = np.ones(len(y)) if w is None else w

    def loss(raw):
        return weights * (raw + y.astype(float) * np.exp(-raw))

    np.testing.assert_allclose(
        actual[:, 0],
        -(loss(a + eps) - loss(a - eps)) / (2 * eps),
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        actual[:, 1],
        -(loss(a + eps) - 2 * loss(a) + loss(a - eps)) / eps**2,
        rtol=1e-4,
        atol=1e-5,
    )
    error, total = gamma_deviance(a, y, w)
    assert error / total == pytest.approx(
        mean_gamma_deviance(y, np.exp(a), sample_weight=w)
    )
    metric = GammaMetric()
    assert metric.evaluate([a], y, w) == pytest.approx((error, total))
    assert metric.get_final_error(error, total) == pytest.approx(error / total)
    assert not metric.is_max_optimal()
    if w is not None:
        scaled = gamma_deviance(a, y, w * 10)
        assert scaled[0] / scaled[1] == pytest.approx(error / total)


def test_gamma_exact_fit_small_errors_and_zero_weight_overflow():
    y = np.array([1e-30, 1.0, 1e30])
    a = np.log(y)
    np.testing.assert_array_equal(gamma_derivatives(a, y, None)[:, 0], 0)
    assert gamma_deviance(a, y, None) == (0, 3)
    z = 1e-7
    error, _ = gamma_deviance(np.array([-z]), np.ones(1), None)
    assert error == pytest.approx(z * z + z**3 / 3, rel=1e-8)
    raw = np.array([-1000.0, 0.0])
    w = np.array([0.0, 1.0])
    np.testing.assert_array_equal(
        gamma_derivatives(raw, np.ones(2), w), [[0, 0], [0, -1]]
    )
    assert gamma_deviance(raw, np.ones(2), w) == (0, 1)


@pytest.mark.parametrize("raw", [-1000.0, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("function", [gamma_derivatives, gamma_deviance])
def test_gamma_detects_nonfinite_computation(raw, function):
    with pytest.raises(ValueError, match="finite"):
        function(np.array([raw]), np.ones(1), None)


@pytest.mark.parametrize(
    "factory", [CatBoostRegressionModel, CatBoostLogLinkRegressionModel]
)
@pytest.mark.parametrize("alias", ["loss", "loss_function"])
def test_gamma_fit_matches_reference_and_preserves_params(data, factory, alias):
    train, val = data.take(range(32)), data.take(range(32, 48))
    params = {
        **PARAMS,
        alias: "Gamma",
        "iterations": 25,
        "early_stopping_rounds": 4,
        "use_best_model": True,
    }
    with warnings.catch_warnings(record=True) as notices:
        model = factory(params).fit(train, val)
    assert not any("optim" in str(w.message) for w in notices)
    assert params == model.params and params[alias] == "Gamma"
    assert isinstance(model.native_model.get_params()["loss_function"], GammaObjective)
    baseline = model.baseline
    if factory is CatBoostLogLinkRegressionModel:
        assert baseline == pytest.approx(
            np.log(np.average(train.y, weights=train.weight))
        )
    else:
        assert baseline is None
    native_params = {key: value for key, value in params.items() if key != alias}
    reference = catboost.CatBoostRegressor(
        **native_params,
        loss_function=ReferenceGammaObjective(),
        eval_metric=GammaMetric(),
        boosting_type="Plain",
        leaf_estimation_method="Newton",
        leaf_estimation_iterations=1,
        boost_from_average=False,
    )
    reference.fit(
        train.to_catboost(baseline=baseline),
        eval_set=val.to_catboost(baseline=baseline),
    )
    if baseline is not None:
        scale, bias = reference.get_scale_and_bias()
        reference.set_scale_and_bias(scale, bias + baseline)
    expected = reference.predict(data.to_model_input(), prediction_type="Exponent")
    np.testing.assert_allclose(model.predict_pool(data), expected, rtol=1e-10)
    np.testing.assert_allclose(model.predict_frame(data.X), expected, rtol=1e-10)
    assert model.best_iteration == reference.get_best_iteration()
    for split in ("learn", "validation"):
        np.testing.assert_allclose(
            model.native_model.get_evals_result()[split]["GammaMetric"],
            reference.get_evals_result()[split]["GammaMetric"],
            rtol=1e-10,
        )
    refit = model.get_refit_params()
    assert refit[alias] == "Gamma" and "early_stopping_rounds" not in refit
    assert refit["iterations"] == model.best_iteration + 1
    assert np.isfinite(factory(refit).fit(data).predict_pool(data)).all()


@pytest.mark.parametrize(
    "factory", [CatBoostRegressionModel, CatBoostLogLinkRegressionModel]
)
@pytest.mark.parametrize("output", [None, "Exponent", "RawFormulaVal"])
def test_gamma_native_and_state_roundtrip(data, tmp_path, factory, output):
    model = factory({**PARAMS, "loss": "Gamma"}, prediction_type=output).fit(data)
    expected = model.predict_pool(data)
    raw = model.native_model.predict(
        data.to_model_input(), prediction_type="RawFormulaVal"
    )
    np.testing.assert_allclose(
        expected, raw if output == "RawFormulaVal" else np.exp(raw)
    )
    path = tmp_path / "gamma.cbm"
    model.native_model.save_model(str(path))
    native = catboost.CatBoostRegressor().load_model(str(path))
    np.testing.assert_allclose(
        native.predict(data.to_model_input(), prediction_type=output or "Exponent"),
        expected,
    )
    state = json.loads(json.dumps(model.export_state()))
    assert state["prediction_type"] == output
    assert state["effective_prediction_type"] == (output or "Exponent")
    assert state["training_params"]["loss"] == "Gamma"
    restored = factory()
    restored.restore_state(state, str(path))
    np.testing.assert_allclose(restored.predict_pool(data), expected)
    restored.fit(data)
    np.testing.assert_allclose(restored.predict_pool(data), expected)


def test_old_regression_state_still_loads(data, tmp_path):
    model = CatBoostLogLinkRegressionModel(PARAMS).fit(data)
    path = tmp_path / "old.cbm"
    model.native_model.save_model(str(path))
    state = model.export_state()
    del state["effective_prediction_type"]
    restored = CatBoostLogLinkRegressionModel()
    restored.restore_state(state, str(path))
    np.testing.assert_allclose(restored.predict_pool(data), model.predict_pool(data))


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), 1e-50, 1e40])
@pytest.mark.parametrize("validation", [False, True])
@pytest.mark.parametrize(
    "factory", [CatBoostRegressionModel, CatBoostLogLinkRegressionModel]
)
def test_gamma_rejects_invalid_targets_even_zero_weight(
    data, value, validation, factory
):
    y, w = data.y.to_numpy().copy(), data.weight.to_numpy().copy()
    y[0], w[0] = value, 0
    invalid = replace(data, y=pl.Series("target", y), weight=pl.Series("weight", w))
    with pytest.raises(ValueError, match="Gamma.*targets.*positive"):
        factory({**PARAMS, "loss_function": "Gamma"}).fit(
            data if validation else invalid, invalid if validation else None
        )


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), 1e-50, 1e40])
@pytest.mark.parametrize("validation", [False, True])
def test_gamma_rejects_invalid_weights(data, value, validation):
    invalid = replace(data, weight=pl.Series("weight", np.full(len(data), value)))
    with pytest.raises(ValueError, match="Gamma.*weights"):
        CatBoostLogLinkRegressionModel({**PARAMS, "loss": "Gamma"}).fit(
            data if validation else invalid, invalid if validation else None
        )


def test_gamma_parameter_defaults_and_explicit_overrides(data):
    with pytest.raises(ValueError, match="CatBoostLogLinkRegressionModel"):
        CatBoostRegressionModel(
            {**PARAMS, "loss": "Gamma", "boost_from_average": True}
        ).fit(data)
    model = CatBoostLogLinkRegressionModel(
        {
            **PARAMS,
            "loss": "Gamma",
            "boost_from_average": True,
            "eval_metric": "Gamma",
            "leaf_estimation_method": "Gradient",
            "leaf_estimation_iterations": 2,
        }
    ).fit(data)
    native = model.native_model.get_all_params()
    assert native["boost_from_average"] is False
    assert native["leaf_estimation_method"] == "Gradient"
    assert native["leaf_estimation_iterations"] == 2
    other = CatBoostLogLinkRegressionModel(
        {**PARAMS, "loss": "Gamma", "eval_metric": "RMSE"}
    ).fit(data)
    assert other.native_model.get_all_params()["eval_metric"] == "RMSE"


def test_refitting_with_different_loss_resets_effective_prediction_type(data):
    model = CatBoostRegressionModel({**PARAMS, "loss": "Gamma"}).fit(data)
    model.params["loss"] = "RMSE"
    model.fit(data)
    assert model.export_state()["effective_prediction_type"] is None
    np.testing.assert_allclose(
        model.predict_pool(data), model.native_model.predict(data.to_model_input())
    )


def test_gamma_early_stopping_uses_validation_deviance(data):
    train = data.take(range(32))
    val = data.take(range(32, 48)).with_target(pl.Series("target", np.full(16, 0.1)))
    model = CatBoostLogLinkRegressionModel(
        {
            **PARAMS,
            "loss": "Gamma",
            "iterations": 100,
            "early_stopping_rounds": 3,
            "use_best_model": True,
        }
    ).fit(train, val)
    history = model.native_model.get_evals_result()["validation"]["GammaMetric"]
    assert len(history) < 100
    assert model.best_iteration == int(np.argmin(history))
    assert model.native_model.tree_count_ == model.best_iteration + 1
    assert mean_gamma_deviance(
        val.y, model.predict_pool(val), sample_weight=val.weight
    ) == pytest.approx(
        min(history),
        rel=1e-6,
    )


def test_pu_metric_imports_remain_compatible():
    from ml_toolkit.models.catboost_classification import UPUMetric, NNPUMetric
    from ml_toolkit.models.losses.pu import common

    assert UPUMetric is common.UPUMetric
    assert NNPUMetric is common.NNPUMetric


def test_gamma_gpu_routing(data, monkeypatch):
    import ml_toolkit.models.catboost_regression as module

    captured = {}

    def capture(model_type, params, train_data, validation_data):
        captured.update(params)
        raise RuntimeError("reached native fit")

    monkeypatch.setattr(module, "native_fit", capture)
    params = {"loss_function": "Gamma", "task_type": "GPU"}
    model = CatBoostLogLinkRegressionModel(params)
    with pytest.raises(RuntimeError, match="reached native fit"):
        model.fit(data)
    assert captured["devices"] == "0"
    assert type(captured["loss_function"]).__name__ == "GPUGammaObjective"
    assert isinstance(captured["eval_metric"], GammaMetric)
    assert model.params == params and "devices" not in params


@pytest.mark.parametrize(
    "params", [{"task_type": "TPU"}, {"task_type": "GPU", "devices": "0:1"}]
)
def test_gamma_invalid_device_parameters(data, params):
    with pytest.raises(ValueError, match="Gamma.*(task_type|one device)"):
        CatBoostLogLinkRegressionModel({"loss": "Gamma", **params}).fit(data)


def test_gamma_cuda_type_inference(monkeypatch):
    from numba import types
    from numba.core.compiler import run_frontend
    from numba.core.target_extension import target_override
    from numba.core.typed_passes import type_inference_stage
    from numba.cuda.descriptor import cuda_target
    from ml_toolkit.models.losses.gamma.cuda import make_gamma_gpu_objective

    monkeypatch.setattr(
        "numba.cuda.dispatcher.get_current_device",
        lambda: SimpleNamespace(compute_capability=(7, 5)),
    )
    method = type(make_gamma_gpu_objective()).calc_ders_range_gpu
    array = types.Array(types.float32, 1, "C")
    with target_override("cuda"):
        cuda_target.typing_context.refresh()
        cuda_target.target_context.refresh()
        result = type_inference_stage(
            cuda_target.typing_context,
            cuda_target.target_context,
            run_frontend(method),
            (types.int64,) + (array,) * 6,
            types.void,
        )
    assert result.return_type == types.void and not result.typing_errors


def test_gamma_cuda_simulator():
    script = Path(__file__).resolve().parents[1] / "gamma_cuda_check.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        env={**os.environ, "NUMBA_ENABLE_CUDASIM": "1"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.gpu
def test_gamma_training_on_cuda_hardware(data, tmp_path):
    from catboost.utils import get_gpu_device_count
    from numba import cuda

    try:
        count = get_gpu_device_count()
    except catboost.CatBoostError:
        count = 0
    if not count or not cuda.is_available():
        pytest.skip(
            "A CUDA device, compatible driver and Numba CUDA toolkit are required"
        )
    train, val = data.take(range(32)), data.take(range(32, 48))
    params = {
        **PARAMS,
        "loss": "Gamma",
        "task_type": "GPU",
        "devices": "0",
        "use_best_model": False,
    }
    model = CatBoostLogLinkRegressionModel(params).fit(train, val)
    predictions = model.predict_pool(val)
    assert np.isfinite(predictions).all() and (predictions > 0).all()
    expected = mean_gamma_deviance(val.y, predictions, sample_weight=val.weight)
    assert model.native_model.get_evals_result()["validation"]["GammaMetric"][
        -1
    ] == pytest.approx(expected, rel=1e-4, abs=1e-5)
    path = tmp_path / "gpu.cbm"
    model.native_model.save_model(str(path))
    restored = CatBoostLogLinkRegressionModel()
    restored.restore_state(model.export_state(), str(path))
    np.testing.assert_allclose(restored.predict_pool(val), predictions)
