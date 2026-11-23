"""nnPU boosting loop: incremental logits, equivalence, warnings, aliases."""

import sys
import warnings

import numpy as np
import polars as pl
import pytest

from ml_toolkit.models import CatBoostPUClassificationModel
from tests.models.support import PARAMS


def _model(extra=None, **overrides):
    params = {
        **PARAMS,
        "iterations": 8,
        "loss": "nnPU",
        "pu_prior": 0.4,
        **(extra or {}),
    }
    for key, value in overrides.items():
        if value is None:
            params.pop(key)
        else:
            params[key] = value
    return CatBoostPUClassificationModel(params)


@pytest.fixture
def split(data):
    binary = data.with_target(pl.Series("target", [0, 1] * 24))
    return binary, binary.take(range(32)), binary.take(range(32, 48))


def test_incremental_logits_match_full_prediction(monkeypatch, split):
    import ml_toolkit.models.losses.pu as module

    _, train, validation = split
    train_calls, validation_calls = [], []
    original_risks = module.pu_risks

    # Train (32 rows) and validation (16 rows) differ in size, so the row
    # count identifies which sample a risk call belongs to.
    # The metric callback shares pu_risks; only adapter and loop calls are recorded.
    def record(logits, labels, weights, prior):
        if sys._getframe(1).f_code.co_name not in (
            "fit_nnpu",
            "_validate_params",
            "_fit_backend",
        ):
            return original_risks(logits, labels, weights, prior)
        calls = train_calls if len(labels) == len(train.y) else validation_calls
        calls.append(np.array(logits))
        return original_risks(logits, labels, weights, prior)

    monkeypatch.setattr(module, "pu_risks", record)
    model = _model({"use_best_model": False}).fit(train, validation)
    native = model.native_model

    def full(pool, trees):
        return native.predict(
            pool.to_catboost(), prediction_type="RawFormulaVal", ntree_end=trees
        )

    # 8 trees: train risk is computed before each tree (8) plus once in fit
    # validation; validation risk once before the loop and after each tree.
    # Calls before the loop use baseline-only logits. Training risk
    # is then evaluated before each tree, validation risk after it.
    assert len(train_calls) == 9 and len(validation_calls) == 10
    for trees, logits in enumerate(train_calls[1:]):
        expected = train_calls[1] + (full(train, trees) if trees else 0.0)
        np.testing.assert_allclose(logits, expected, atol=1e-9)
    for trees, logits in enumerate(validation_calls[2:], start=1):
        np.testing.assert_allclose(
            logits, validation_calls[0] + full(validation, trees), atol=1e-9
        )


def test_fit_emits_no_numba_warnings(split):
    _, train, validation = split
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _model().fit(train, validation)


def test_num_boost_round_alias(split):
    _, train, validation = split
    model = _model(iterations=None, num_boost_round=3).fit(train, validation)
    assert model.native_model.tree_count_ == 3
    with pytest.raises(ValueError, match="Specify only one"):
        _model(num_boost_round=3).fit(train, validation)
