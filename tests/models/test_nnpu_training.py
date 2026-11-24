"""nnPU in one CatBoost fit: per-tree equivalence, early stopping, validation."""

import warnings
from datetime import date, timedelta

import catboost
import numpy as np
import polars as pl
import pytest

from examples.project_pool import ProjectPool as Pool
from ml_toolkit.models import CatBoostClassificationModel, CatBoostPUClassificationModel
from ml_toolkit.models.losses.pu import (
    NNPUCorrection,
    NNPUMetric,
    NNPUObjective,
    pu_risks,
)
from tests.models.support import PARAMS

CLASS_PRIOR = 0.4
SMALL_PARAMS = {**PARAMS, "iterations": 8, "loss": "nnPU", "class_prior": 0.7}


@pytest.fixture
def split(data):
    binary = data.with_target(pl.Series("target", [0, 1] * 24))
    return binary, binary.take(range(32)), binary.take(range(32, 48))


@pytest.fixture
def noisy_pool():
    rng = np.random.default_rng(7)
    size = 400
    X = rng.normal(size=(size, 4))
    labeled = (X[:, 0] + 0.8 * rng.normal(size=size) > 1.0).astype(int)
    return Pool(
        X=pl.DataFrame({f"f{i}": X[:, i] for i in range(4)}),
        y=pl.Series("target", labeled),
        time=pl.Series(
            "date", [date(2024, 1, 1) + timedelta(days=i) for i in range(size)]
        ),
        weight=pl.Series("weight", rng.uniform(0.5, 2.0, size)),
        cat_features=(),
    )


def _reference_logits(train, baseline, trees, native_params, beta, gamma):
    """Per-tree loop: decide the branch from full training logits, fit one tree."""
    labels, weights = train.y.to_numpy(), train.weight.to_numpy()
    positive_scale = CLASS_PRIOR * weights.sum() / weights[labels == 1].sum()
    pool = train.to_catboost(baseline=baseline)
    plain = train.to_catboost()
    correction = NNPUCorrection(beta, gamma, len(train))
    logits = np.full(len(train), baseline)
    model, reversed_trees = None, 0
    for _ in range(trees):
        _, negative = pu_risks(logits, labels, weights, CLASS_PRIOR)
        correction.reverse = negative < -beta
        reversed_trees += correction.reverse
        step = catboost.CatBoostClassifier(
            **native_params,
            iterations=1,
            loss_function=NNPUObjective(positive_scale, correction),
            eval_metric=NNPUMetric(CLASS_PRIOR),
            leaf_estimation_method="Gradient",
            leaf_estimation_iterations=1,
            boosting_type="Plain",
            boost_from_average=False,
        )
        step.fit(pool, init_model=model)
        model = step
        logits = baseline + model.predict(plain, prediction_type="RawFormulaVal")
    return model, reversed_trees


@pytest.mark.parametrize(
    "beta, gamma", [(None, None), (0.05, 0.5)], ids=["defaults", "beta-gamma"]
)
@pytest.mark.parametrize("with_validation", [False, True])
def test_single_fit_equals_per_tree_reference_loop(
    noisy_pool, beta, gamma, with_validation
):
    trees = 40
    train = noisy_pool.take(range(200))
    validation = noisy_pool.take(range(200, 400)) if with_validation else None
    native_params = dict(
        depth=6,
        learning_rate=0.5,
        random_seed=3,
        thread_count=1,
        verbose=False,
        allow_writing_files=False,
    )
    extra = {}
    if beta is not None:
        extra = {"nnpu_beta": beta, "nnpu_gamma": gamma}
    model = CatBoostPUClassificationModel(
        {
            **native_params,
            "iterations": trees,
            "loss": "nnPU",
            "class_prior": CLASS_PRIOR,
            "use_best_model": False,
            **extra,
        }
    ).fit(train, validation)
    reference, reversed_trees = _reference_logits(
        train,
        model.baseline,
        trees,
        native_params,
        0.0 if beta is None else beta,
        0.2 if gamma is None else gamma,  # documented default
    )
    # The comparison is only meaningful when the correction actually triggers.
    assert reversed_trees >= 1
    features = noisy_pool.to_model_input()
    np.testing.assert_allclose(
        model.native_model.predict(features, prediction_type="RawFormulaVal"),
        reference.predict(features, prediction_type="RawFormulaVal"),
        rtol=0,
        atol=1e-9,
    )


def test_early_stopping_is_native(split):
    _, train, validation = split
    model = CatBoostPUClassificationModel(
        {**SMALL_PARAMS, "iterations": 200, "early_stopping_rounds": 2}
    ).fit(train, validation)
    assert model.best_iteration is not None
    assert model.native_model.tree_count_ == model.best_iteration + 1
    assert model.native_model.tree_count_ < 200


def test_fit_emits_no_numba_warnings(split):
    _, train, validation = split
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        CatBoostPUClassificationModel(SMALL_PARAMS).fit(train, validation)


def test_num_boost_round_alias(split):
    _, train, validation = split
    params = {k: v for k, v in SMALL_PARAMS.items() if k != "iterations"}
    model = CatBoostPUClassificationModel({**params, "num_boost_round": 3}).fit(
        train, validation
    )
    assert model.native_model.tree_count_ == 3


def test_missing_learn_evaluation_raises(monkeypatch, split):
    _, train, validation = split

    def skip_update(self, approxes, target, weight):
        positive, negative = self._components(approxes, target, weight)
        return positive + max(0.0, negative), 1.0

    monkeypatch.setattr(NNPUMetric, "evaluate", skip_update)
    with pytest.raises(RuntimeError, match="learn-set nnPU risk"):
        CatBoostPUClassificationModel(SMALL_PARAMS).fit(train, validation)


@pytest.mark.parametrize(
    "extra, message",
    [
        ({"pu_prior": 0.7}, "class_prior"),
        ({"class_prior": None}, "requires class_prior"),
        ({"class_prior": 0.3}, "below the weighted share"),
        ({"nnpu_beta": -0.1}, "nnpu_beta"),
        ({"nnpu_beta": float("nan")}, "nnpu_beta"),
        ({"nnpu_beta": True}, "nnpu_beta"),
        ({"nnpu_gamma": 0.0}, "nnpu_gamma"),
        ({"nnpu_gamma": 1.5}, "nnpu_gamma"),
        ({"nnpu_gamma": float("nan")}, "nnpu_gamma"),
        ({"nnpu_gamma": True}, "nnpu_gamma"),
        ({"eval_metric": "Logloss"}, "own risk as eval_metric"),
        ({"metric_period": 2}, "metric_period must be 1"),
    ],
)
def test_nnpu_parameter_validation(split, extra, message):
    _, train, _ = split
    params = {**SMALL_PARAMS, **extra}
    if params.get("class_prior", 0) is None:
        del params["class_prior"]
    with pytest.raises(ValueError, match=message):
        CatBoostPUClassificationModel(params).fit(train)


def test_nnpu_options_require_nnpu_loss(split):
    _, train, _ = split
    params = {**SMALL_PARAMS, "loss": "uPU", "nnpu_beta": 0.1}
    with pytest.raises(ValueError, match="require loss='nnPU'"):
        CatBoostPUClassificationModel(params).fit(train)


@pytest.mark.parametrize("name", ["class_prior", "nnpu_beta", "nnpu_gamma"])
def test_pu_options_are_rejected_by_plain_classifier(split, name):
    _, train, _ = split
    with pytest.raises(ValueError, match=f"{name} requires"):
        CatBoostClassificationModel({**PARAMS, name: 0.5}).fit(train)
