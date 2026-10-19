"""uPU numerical contracts, native training equivalence and CUDA coverage."""

import os
import subprocess
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import catboost
import numpy as np
import polars as pl
import pytest

from examples.project_pool import ProjectPool as Pool
from ml_toolkit.models import CatBoostPUClassificationModel
from ml_toolkit.models._catboost_upu import UPUObjective, upu_derivatives, upu_risk
from ml_toolkit.models.catboost_classification import (
    UPUMetric,
    _PULogisticObjective,
    _PURiskMetric,
    _pu_risks,
)


@pytest.fixture
def pu_data():
    rng = np.random.default_rng(230)
    size = 120
    labels = np.tile([1, 0, 0, 1, 0], size // 5)
    return Pool(
        X=pl.DataFrame(
            {
                "signal": labels + rng.normal(size=size),
                "noise": rng.normal(size=size),
                "category": np.tile(["a", "b", "c"], size // 3),
            }
        ),
        y=pl.Series("target", labels),
        time=pl.Series(
            "date", [date(2024, 1, 1) + timedelta(days=i) for i in range(size)]
        ),
        weight=pl.Series("weight", rng.uniform(0.1, 3, size)),
        cat_features=("category",),
    )


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_compiled_cpu_derivatives_match_reference_and_finite_differences(
    weighted, dtype
):
    logits = np.array([-1000, 1000, -12, 0, 5, -5, 0.7, 1000, -1000], dtype=float)
    labels = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0], dtype=dtype)
    weights = (
        np.array([1, 0, 0.2, 1.2, 3, 0, 2, 1, 0.5], dtype=dtype) if weighted else None
    )
    positive_scale, unlabeled_scale = 0.73, 1.89
    for array in (logits, labels, weights):
        if array is not None:
            array.setflags(write=False)
    result = upu_derivatives(logits, labels, weights, positive_scale, unlabeled_scale)
    reference = _PULogisticObjective(0.2, positive_scale, unlabeled_scale)
    np.testing.assert_allclose(
        result, reference.calc_ders_range(logits, labels, weights), atol=1e-15
    )
    assert result.shape == (len(logits), 2) and result.dtype == np.float64
    assert np.isfinite(result).all()

    effective_weights = np.ones(len(labels)) if weights is None else weights

    def loss(z):
        return effective_weights * np.where(
            labels == 1, -positive_scale * z, unlabeled_scale * np.logaddexp(0, z)
        )

    epsilon = 1e-4
    numerical = (loss(logits + epsilon) - loss(logits - epsilon)) / (2 * epsilon)
    np.testing.assert_allclose(result[:, 0], -numerical, atol=1e-8)
    # A callback sees chunks of a pool: its normalization must not change.
    chunked = np.concatenate(
        [
            upu_derivatives(
                logits[s],
                labels[s],
                None if weights is None else weights[s],
                positive_scale,
                unlabeled_scale,
            )
            for s in (slice(0, 2), slice(2, 6), slice(6, None))
        ]
    )
    np.testing.assert_array_equal(chunked, result)


@pytest.mark.parametrize("weighted", [False, True])
def test_compiled_risk_uses_each_pool_group_normalization(weighted):
    rng = np.random.default_rng(60)
    for size, prior in [(27, 0.03), (64, 0.75)]:
        logits = rng.uniform(-1000, 1000, size)
        labels = (np.arange(size) % 3 == 0).astype(np.float32)
        weights = rng.uniform(0.1, 2, size).astype(np.float32) if weighted else None
        expected = sum(
            _pu_risks(
                logits, labels, np.ones(size) if weights is None else weights, prior
            )
        )
        assert upu_risk(logits, labels, weights, prior) == pytest.approx(
            expected, rel=1e-6
        )
        metric = UPUMetric(prior)
        assert metric.evaluate([logits], labels, weights) == pytest.approx(
            (expected, 1), rel=1e-6
        )
        if weights is not None:
            assert upu_risk(logits, labels, weights * 11, prior) == pytest.approx(
                expected, rel=1e-6
            )


@pytest.mark.parametrize("logit", [float("inf"), float("nan")])
def test_compiled_risk_rejects_nonfinite_predictions(logit):
    with pytest.raises(ValueError, match="finite"):
        upu_risk(np.array([logit, 0.0]), np.array([1.0, 0.0]), None, 0.2)


def test_compiled_cpu_fit_preserves_predictions_validation_and_baseline(pu_data):
    train, val = pu_data.take(range(80)), pu_data.take(range(80, 120))
    prior = 0.23
    params = dict(
        iterations=25,
        depth=3,
        learning_rate=0.07,
        random_seed=51,
        thread_count=2,
        verbose=False,
        allow_writing_files=False,
        use_best_model=True,
        early_stopping_rounds=7,
    )
    with warnings.catch_warnings(record=True) as notices:
        accelerated = CatBoostPUClassificationModel(
            {**params, "loss": "uPU", "pu_prior": prior}
        ).fit(train, val)
    assert not any("optim" in str(w.message) for w in notices)
    assert isinstance(
        accelerated.native_model.get_params()["loss_function"], UPUObjective
    )
    y, w = train.y.to_numpy(), train.weight.to_numpy()
    objective = _PULogisticObjective(
        prior,
        float(prior * w.sum() / w[y == 1].sum()),
        float(w.sum() / w[y == 0].sum()),
    )
    reference = catboost.CatBoostClassifier(
        **params,
        loss_function=objective,
        eval_metric=_PURiskMetric(prior, non_negative=False),
        leaf_estimation_method="Gradient",
        leaf_estimation_iterations=1,
        boosting_type="Plain",
        boost_from_average=False,
    )
    baseline = accelerated.baseline
    reference.fit(
        train.to_catboost(baseline=baseline),
        eval_set=val.to_catboost(baseline=baseline),
    )
    np.testing.assert_allclose(
        accelerated.native_model.predict(
            pu_data.to_model_input(), prediction_type="RawFormulaVal"
        ),
        reference.predict(pu_data.to_model_input(), prediction_type="RawFormulaVal"),
        rtol=1e-12,
        atol=1e-12,
    )
    reference_logits = (
        reference.predict(pu_data.to_model_input(), prediction_type="RawFormulaVal")
        + baseline
    )
    np.testing.assert_allclose(
        accelerated.predict_pool(pu_data), 1 / (1 + np.exp(-reference_logits))
    )
    assert accelerated.best_iteration == reference.get_best_iteration()
    for split in ("learn", "validation"):
        np.testing.assert_allclose(
            accelerated.native_model.get_evals_result()[split]["UPUMetric"],
            reference.get_evals_result()[split]["_PURiskMetric"],
            rtol=1e-12,
            atol=1e-12,
        )


def test_gpu_routing_uses_cuda_objective_without_changing_user_params(
    pu_data, monkeypatch
):
    import ml_toolkit.models.catboost_classification as module

    captured = {}

    def capture(model_type, params, train_data, validation_data):
        captured.update(params)
        raise RuntimeError("reached native fit")

    monkeypatch.setattr(module, "native_fit", capture)
    params = {"loss": "uPU", "pu_prior": 0.2, "task_type": "GPU"}
    model = CatBoostPUClassificationModel(params)
    with pytest.raises(RuntimeError, match="reached native fit"):
        model.fit(pu_data)
    assert captured["devices"] == "0"
    assert type(captured["loss_function"]).__name__ == "GPUUPUObjective"
    assert isinstance(captured["eval_metric"], UPUMetric)
    assert model.params == params and "devices" not in params


def test_cuda_kernel_type_inference_without_device(monkeypatch):
    from numba import types
    from numba.core.compiler import run_frontend
    from numba.core.target_extension import target_override
    from numba.core.typed_passes import type_inference_stage
    from numba.cuda.descriptor import cuda_target

    from ml_toolkit.models._catboost_upu_cuda import make_upu_gpu_objective

    # Exercise the actual CUDA type checker; the simulator runs Python and
    # cannot detect unsupported overloads such as the variadic max(a, b).
    # Device overload compilation consults the architecture even before it
    # checks argument signatures, so supply it without requiring a GPU.
    monkeypatch.setattr(
        "numba.cuda.dispatcher.get_current_device",
        lambda: SimpleNamespace(compute_capability=(7, 5)),
    )
    method = type(make_upu_gpu_objective(0.71, 1.37)).calc_ders_range_gpu
    array = types.Array(types.float32, 1, "C")
    # CatBoost supplies a dummy self argument followed by six float32 arrays.
    args = (types.int64,) + (array,) * 6
    with target_override("cuda"):
        cuda_target.typing_context.refresh()
        cuda_target.target_context.refresh()
        result = type_inference_stage(
            cuda_target.typing_context,
            cuda_target.target_context,
            run_frontend(method),
            args,
            types.void,
        )
    assert result.return_type == types.void
    assert not result.typing_errors


def test_cuda_kernel_under_numba_simulator():
    # A subprocess keeps simulator mode isolated from real CUDA integration tests.
    script = Path(__file__).with_name("upu_cuda_check.py")
    env = {**os.environ, "NUMBA_ENABLE_CUDASIM": "1"}
    result = subprocess.run(
        [sys.executable, str(script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_upu_gpu_training_on_cuda_hardware(pu_data, tmp_path):
    from catboost.utils import get_gpu_device_count

    try:
        devices = get_gpu_device_count()
    except catboost.CatBoostError:
        devices = 0
    if devices == 0:
        pytest.skip("A CUDA device is required for CatBoost GPU integration")
    from numba import cuda

    if not cuda.is_available():
        pytest.skip("Numba CUDA and a compatible CUDA toolkit are required")
    train, val = pu_data.take(range(80)), pu_data.take(range(80, 120))
    model = CatBoostPUClassificationModel(
        {
            "loss": "uPU",
            "pu_prior": 0.23,
            "task_type": "GPU",
            "devices": "0",
            "iterations": 8,
            "depth": 2,
            "learning_rate": 0.05,
            "verbose": False,
            "allow_writing_files": False,
            "use_best_model": False,
        }
    ).fit(train, val)
    predictions = model.predict_pool(val)
    assert (
        np.isfinite(predictions).all()
        and ((predictions >= 0) & (predictions <= 1)).all()
    )
    logits = (
        model.native_model.predict(
            val.to_model_input(), prediction_type="RawFormulaVal"
        )
        + model.baseline
    )
    expected = upu_risk(logits, val.y.to_numpy(), val.weight.to_numpy(), 0.23)
    assert model.native_model.get_evals_result()["validation"]["UPUMetric"][
        -1
    ] == pytest.approx(expected, rel=1e-4, abs=1e-5)
    path = tmp_path / "upu.cbm"
    model.native_model.save_model(str(path))
    restored = catboost.CatBoostClassifier().load_model(str(path))
    restored_logits = (
        restored.predict(val.to_model_input(), prediction_type="RawFormulaVal")
        + model.baseline
    )
    np.testing.assert_allclose(1 / (1 + np.exp(-restored_logits)), predictions)
