"""Native initialization, classifier thresholds, and PU training policies."""

import catboost
import numpy as np
import polars as pl
import pytest
from sklearn.metrics import f1_score

from ml_toolkit.models import (
    CatBoostClassificationModel,
    CatBoostLogLinkRegressionModel,
    CatBoostPUClassificationModel,
    CatBoostRegressionModel,
)
from ml_toolkit.training.evaluation import calculate_metrics
from tests.models.support import PARAMS, SquaredObjective
from tests.support.training import mae


@pytest.mark.parametrize(
    "loss", [None, "RMSE", "MAE", "Quantile:alpha=0.7", SquaredObjective()]
)
def test_general_regressor_preserves_native_initialization_and_outputs(data, loss):
    data = data.with_target(data.y - 8)
    params = {**PARAMS, "boost_from_average": False}
    if loss is not None:
        params["loss"] = loss
    if isinstance(loss, SquaredObjective):
        params["eval_metric"] = "RMSE"
        params["leaf_estimation_method"] = "Gradient"
    model = CatBoostRegressionModel(params).fit(data)
    native_params = {key: value for key, value in params.items() if key != "loss"}
    if loss is not None:
        native_params["loss_function"] = loss
    native = catboost.CatBoostRegressor(**native_params).fit(data.to_catboost())
    assert model.baseline is None
    assert model.task == "regression"
    assert model.native_model.get_params()["boost_from_average"] is False
    np.testing.assert_allclose(
        model.predict_pool(data), native.predict(data.to_model_input())
    )
    assert (model.predict_pool(data) < 0).any()
    assert params == model.params


def test_native_boost_from_average_is_preserved(data):
    model = CatBoostRegressionModel({**PARAMS, "boost_from_average": True}).fit(
        data.with_target(data.y - 10)
    )
    assert model.baseline is None
    assert model.native_model.get_all_params()["boost_from_average"] is True
    assert model.native_model.get_scale_and_bias()[1] != 0


@pytest.mark.parametrize("loss", ["Poisson", "Tweedie:variance_power=1.5"])
def test_log_link_baseline_shared_by_splits_and_folded_once(data, loss):
    train, validation = data.take(range(32)), data.take(range(32, 48))
    baseline = float(
        np.log(np.average(train.y.to_numpy(), weights=train.weight.to_numpy()))
    )
    model = CatBoostLogLinkRegressionModel(
        {**PARAMS, "loss": loss, "boost_from_average": True}
    ).fit(train, validation)
    native = catboost.CatBoostRegressor(
        **PARAMS, loss_function=loss, boost_from_average=False
    )
    native.fit(
        train.to_catboost(baseline=baseline),
        eval_set=validation.to_catboost(baseline=baseline),
    )
    scale, bias = native.get_scale_and_bias()
    assert model.baseline == pytest.approx(baseline)
    assert model.native_model.get_scale_and_bias() == pytest.approx(
        (scale, bias + baseline)
    )
    native.set_scale_and_bias(scale, bias + baseline)
    np.testing.assert_allclose(
        model.predict_pool(data), native.predict(data.to_model_input())
    )
    assert model.native_model.get_all_params()["boost_from_average"] is False


@pytest.mark.parametrize("loss", ["RMSE", "MAE", SquaredObjective()])
def test_loss_restriction_belongs_only_to_log_link(data, loss):
    with pytest.raises(ValueError, match="Poisson or Tweedie"):
        CatBoostLogLinkRegressionModel({**PARAMS, "loss": loss}).fit(data)


@pytest.mark.parametrize("target", [0.0, -1.0, float("nan")])
def test_log_link_rejects_invalid_weighted_mean(data, target):
    with pytest.raises(ValueError, match="finite and positive"):
        CatBoostLogLinkRegressionModel(PARAMS).fit(
            data.with_target(np.full(len(data), target))
        )


@pytest.mark.parametrize(
    "factory",
    [
        CatBoostRegressionModel,
        CatBoostLogLinkRegressionModel,
        CatBoostClassificationModel,
        CatBoostPUClassificationModel,
    ],
)
def test_loss_aliases_reject_conflicts(factory, data):
    with pytest.raises(ValueError, match="Conflicting"):
        factory({**PARAMS, "loss": "Poisson", "loss_function": "RMSE"}).fit(data)


def test_prediction_type_and_multidimensional_output_are_preserved(data):
    model = CatBoostRegressionModel(PARAMS, prediction_type="RawFormulaVal").fit(data)
    np.testing.assert_allclose(
        model.predict_pool(data),
        model.native_model.predict(
            data.to_model_input(), prediction_type="RawFormulaVal"
        ),
    )

    class Native:
        feature_names_ = ["signal"]

        def get_cat_feature_indices(self):
            return []

        def predict(self, frame, **options):
            return np.zeros((len(frame), 2))

    model._model = Native()
    prediction = model.predict_pool(data)
    assert prediction.shape == (len(data), 2)
    with pytest.raises(ValueError, match="one finite value per row"):
        calculate_metrics(mae, data, prediction)


def test_supervised_prevalence_and_population_prior_are_distinct(data):
    data = data.with_target(pl.Series("target", [1, 0, 0] * 16))
    supervised = CatBoostClassificationModel(PARAMS).fit(data)
    pu = CatBoostPUClassificationModel({**PARAMS, "pu_prior": 0.12}).fit(data)
    prevalence = np.average(data.y.to_numpy(), weights=data.weight.to_numpy())
    assert supervised.baseline == pytest.approx(np.log(prevalence / (1 - prevalence)))
    assert pu.baseline == pytest.approx(np.log(0.12 / 0.88))
    assert pu.pu_prior == 0.12 and pu.pu_loss == "uPU"
    assert pu.task == "pu_classification" and supervised.task == "classification"
    with pytest.raises(ValueError, match="CatBoostPUClassificationModel"):
        CatBoostClassificationModel({**PARAMS, "pu_prior": 0.12}).fit(data)
    with pytest.raises(ValueError, match="Logloss or CrossEntropy"):
        CatBoostClassificationModel({**PARAMS, "loss": "uPU"}).fit(data)


@pytest.mark.parametrize("with_validation", [False, True])
@pytest.mark.parametrize("threshold", [None, 0.8])
def test_cross_entropy_soft_targets_keep_default_or_explicit_threshold(
    data, with_validation, threshold
):
    soft = data.with_target(np.linspace(0.1, 0.9, len(data)))
    train, validation = soft.take(range(32)), soft.take(range(32, 48))
    params = {**PARAMS, "loss": "CrossEntropy"}
    if threshold is not None:
        params["decision_threshold"] = threshold
    model = CatBoostClassificationModel(params).fit(
        train, validation if with_validation else None
    )
    expected = 0.5 if threshold is None else threshold
    probabilities = model.predict_pool(validation)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert model.threshold == expected
    assert model.validation_f1 is None
    np.testing.assert_array_equal(
        model.predict_labels_pool(validation), probabilities >= expected
    )
    refitted = CatBoostClassificationModel(model.get_refit_params()).fit(soft)
    assert refitted.threshold == expected
    assert refitted.validation_f1 is None


def test_soft_validation_resets_previous_binary_f1_state(data):
    binary = data.with_target(pl.Series("target", [1, 0, 0] * 16))
    model = CatBoostClassificationModel({**PARAMS, "loss": "CrossEntropy"})
    model.fit(binary.take(range(32)), binary.take(range(32, 48)))
    assert model.validation_f1 is not None
    soft = data.with_target(np.linspace(0.1, 0.9, len(data)))
    model.fit(soft.take(range(32)), soft.take(range(32, 48)))
    assert model.validation_f1 is None
    assert model.threshold == 0.5


@pytest.mark.parametrize("loss", ["Logloss", "CrossEntropy"])
@pytest.mark.parametrize("threshold", [None, 0.8])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_validation_targets_fail_before_native_fit(
    data, loss, threshold, invalid, monkeypatch
):
    binary = data.with_target(pl.Series("target", [1, 0, 0] * 16))
    validation = binary.take(range(32, 48))
    targets = validation.y.to_numpy().astype(float)
    targets[0] = invalid
    validation = validation.with_target(targets)
    params = {**PARAMS, "loss": loss}
    if threshold is not None:
        params["decision_threshold"] = threshold
    monkeypatch.setattr(
        "ml_toolkit.models.catboost_classification.native_fit",
        lambda *args, **kwargs: pytest.fail("Invalid labels reached native training"),
    )
    with pytest.raises(ValueError, match="validation targets must be between 0 and 1"):
        CatBoostClassificationModel(params).fit(binary.take(range(32)), validation)


def test_logloss_rejects_soft_validation_targets(data):
    binary = data.with_target(pl.Series("target", [1, 0, 0] * 16))
    validation = binary.take(range(32, 48)).with_target(np.full(16, 0.2))
    with pytest.raises(ValueError, match="Logloss validation targets must be binary"):
        CatBoostClassificationModel({**PARAMS, "loss": "Logloss"}).fit(
            binary.take(range(32)), validation
        )


@pytest.mark.parametrize("loss", ["Logloss", "CrossEntropy"])
def test_supervised_threshold_policy_and_refit_reset(data, loss):
    binary = data.with_target(pl.Series("target", [1, 0, 0] * 16))
    train, validation = binary.take(range(32)), binary.take(range(32, 48))
    model = CatBoostClassificationModel({**PARAMS, "loss": loss}).fit(train, validation)
    probabilities = model.predict_pool(validation)
    # Use an independent F1 oracle, not the selector being exercised by fit.
    scores = [
        f1_score(
            validation.y.to_numpy(),
            probabilities >= threshold,
            sample_weight=validation.weight.to_numpy(),
            zero_division=0,
        )
        for threshold in np.unique(probabilities)
    ]
    expected_f1 = max(scores)
    assert model.validation_f1 == pytest.approx(expected_f1)
    assert f1_score(
        validation.y.to_numpy(),
        model.predict_labels_pool(validation),
        sample_weight=validation.weight.to_numpy(),
    ) == pytest.approx(expected_f1)
    model.fit(binary)
    assert model.threshold == 0.5 and model.validation_f1 is None
    explicit = CatBoostClassificationModel({**PARAMS, "decision_threshold": 0.8}).fit(
        train, validation
    )
    assert explicit.threshold == 0.8 and explicit.validation_f1 is None


@pytest.mark.parametrize("loss", ["uPU", "nnPU"])
def test_pu_threshold_policy_and_refit(data, loss):
    binary = data.with_target(pl.Series("target", [1, 0, 0] * 16))
    model = CatBoostPUClassificationModel({**PARAMS, "loss": loss, "pu_prior": 0.2})
    model.fit(binary.take(range(32)), binary.take(range(32, 48)))
    assert model.validation_f1 is None
    assert np.isfinite(model.predict_pool(binary)).all()
    with pytest.raises(RuntimeError, match="decision_threshold"):
        model.predict_labels_pool(binary)
    model.params["decision_threshold"] = 0.4
    model.fit(binary)
    assert model.threshold == 0.4 and model.validation_f1 is None
    assert model.best_iteration is None


def test_nnpu_correction_branch_early_stopping_and_best_tree(monkeypatch, data):
    import ml_toolkit.models.catboost_classification as module

    binary = data.with_target(pl.Series("target", [0, 1] * 24))
    calls, risks = [], iter([3.0, 2.0, 4.0, 5.0])
    original_risks = module.pu_risks
    # Control training versus validation separately; retain the actual native fits.
    train, val = binary.take(range(32)), binary.take(range(32, 48))

    def risk(logits, labels, weights, prior):
        original_risks(logits, labels, weights, prior)
        if len(labels) == 32:
            return 1.0, -1.0
        if len(calls) == 0:
            return 1.0, 1.0
        return 0.0, next(risks)

    fit = catboost.CatBoostClassifier.fit

    def record(self, *args, **kwargs):
        objective = self.get_params()["loss_function"]
        calls.append(objective.reverse_negative_risk)
        return fit(self, *args, **kwargs)

    monkeypatch.setattr(module, "pu_risks", risk)
    monkeypatch.setattr(catboost.CatBoostClassifier, "fit", record)
    model = CatBoostPUClassificationModel(
        {
            **PARAMS,
            "iterations": 10,
            "loss": "nnPU",
            "pu_prior": 0.7,
            "early_stopping_rounds": 2,
            "eval_metric": "Logloss",
        }
    ).fit(train, val)
    assert calls == [True] * 4
    assert model.best_iteration == 1 and model.native_model.tree_count_ == 2


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), "invalid"])
@pytest.mark.parametrize(
    "factory, extra",
    [(CatBoostClassificationModel, {}), (CatBoostPUClassificationModel, {"pu_prior": 0.2})],
)
def test_both_classifiers_validate_explicit_threshold(data, threshold, factory, extra):
    binary = data.with_target(pl.Series("target", [1, 0, 0] * 16))
    with pytest.raises(ValueError, match="decision_threshold"):
        factory({**PARAMS, **extra, "decision_threshold": threshold}).fit(binary)


@pytest.mark.parametrize(
    "extra, message",
    [
        ({}, "requires pu_prior"),
        ({"pu_prior": 0.0}, "strictly between"),
        ({"pu_prior": float("nan")}, "strictly between"),
        ({"pu_prior": 0.2, "loss": "nnPU", "task_type": "GPU"}, "CPU"),
        ({"pu_prior": 0.2, "task_type": "TPU"}, "task_type"),
        ({"pu_prior": 0.2, "task_type": "GPU", "devices": "0:1"}, "one device"),
        ({"pu_prior": 0.2, "class_weights": [1, 2]}, "incompatible"),
    ],
)
def test_pu_constraints_are_local_to_pu(data, extra, message):
    binary = data.with_target(pl.Series("target", [1, 0, 0] * 16))
    with pytest.raises(ValueError, match=message):
        CatBoostPUClassificationModel({**PARAMS, **extra}).fit(binary)


def test_pu_objective_derivatives_and_risks_are_stable():
    from ml_toolkit.models.losses.pu.common import (
        pu_risks,
        PULogisticObjective,
    )

    logits = np.array([-1000.0, 1000.0, -1000.0, 1000.0])
    labels = np.array([1, 1, 0, 0])
    weights = np.ones(4)
    objective = PULogisticObjective(0.5, positive_scale=2.0, unlabeled_scale=3.0)
    np.testing.assert_allclose(
        objective.calc_ders_range(logits, labels, weights),
        [[2.0, 0.0], [2.0, 0.0], [0.0, 0.0], [-3.0, 0.0]],
    )
    objective.reverse_negative_risk = True
    np.testing.assert_allclose(
        objective.calc_ders_range(logits, labels, weights),
        [[0.0, 0.0], [-2.0, 0.0], [0.0, 0.0], [3.0, 0.0]],
    )
    assert np.isfinite(pu_risks(logits, labels, weights, 0.5)).all()
