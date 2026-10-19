"""Batch RFE selection, objective handling and adapter/ranker boundaries."""

from dataclasses import dataclass, replace
from datetime import date, timedelta
from types import SimpleNamespace

import catboost
import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

from examples.project_pool import ProjectPool as Pool
from ml_toolkit.feature_selection import RFESelector
from ml_toolkit.config import HoldoutConfig, TimeSeriesCVConfig, TrainingConfig
from ml_toolkit.metrics import MetricScores
from ml_toolkit.models import CatBoostRegressionModel
from ml_toolkit.training import (
    ShapRanker,
    importance_ranker,
    split_holdout,
    train_model,
)


@pytest.fixture
def data():
    count = 24
    return Pool(
        X=pl.DataFrame({f"f{i}": np.arange(count) + i for i in range(6)}),
        y=pl.Series("target", np.ones(count)),
        time=pl.Series(
            "date", [date(2024, 1, 1) + timedelta(days=i) for i in range(count)]
        ),
        weight=pl.Series("weight", np.arange(count, dtype=float) + 1),
        object_id=pl.Series("id", np.arange(count)),
        business_type=pl.Series("group", ["a", "b"] * 12),
        numerical_features=tuple(f"f{i}" for i in range(6)),
    )


def settings(**overrides):
    return RFESelector(
        ranker=importance_ranker,
        **{
            "min_features": 2,
            "metric": "objective",
            "direction": "minimize",
            "max_fits": 3,
            **overrides,
        },
    )


@dataclass
class Scores:
    def to_dict(self):
        return {"objective": float(self.objective), "secondary": float(self.secondary)}

    objective: float
    secondary: float


def scores(target, prediction, weight):
    return Scores(float(np.average(prediction, weights=weight)), float(np.mean(target)))


def ordered_ranker(model, data):
    return {
        name: float(len(data.X.columns) - i) for i, name in enumerate(data.X.columns)
    }


def factory(quality, calls):
    class Adapter:
        best_iteration = 2

        def __init__(self, params):
            self.params = params
            self.params["local"] = True

        def fit(self, train, validation):
            self.names = tuple(train.X.columns)
            calls.append((self, train, validation))
            return self

        def predict_pool(self, data):
            assert tuple(data.X.columns) == self.names
            return np.full(len(data), quality[len(self.names)])

    return Adapter


def run(data, quality, config=None, ranker=ordered_ranker, metric=scores, holdout=None):
    calls = []
    params = {"seed": 123}
    partition = split_holdout(data, holdout or HoldoutConfig(val_size=0.25))
    result = replace(config or settings(), ranker=ranker).select(
        train=partition.train,
        validation=partition.val,
        params=params,
        model_factory=factory(quality, calls),
        metric=metric,
    )
    assert params == {"seed": 123}
    return result, calls


@pytest.mark.parametrize(
    "quality, direction, tolerance, selected",
    [
        ({6: 3, 3: 1, 2: 2}, "minimize", 0, 1),
        ({6: 1, 3: 2, 2: 3}, "minimize", 0, 0),
        ({6: 3, 3: 1, 2: 1.1}, "minimize", 0.11, 2),
        ({6: 1, 3: 3, 2: 2}, "maximize", 0, 1),
        ({6: 1, 3: 3, 2: 2.9}, "maximize", 0.11, 2),
        ({6: -1, 3: -3, 2: -2.9}, "minimize", 0.11, 2),
        ({6: 1, 3: 1, 2: 1}, "minimize", 0, 2),
    ],
)
def test_selects_best_visited_subset_and_tolerance(
    data, quality, direction, tolerance, selected
):
    result, calls = run(
        data, quality, settings(direction=direction, score_tolerance=tolerance)
    )
    assert result.selected_step == selected
    assert result.features == result.steps[selected].features
    assert len(calls) == 3
    assert len({id(model) for model, _, _ in calls}) == 3
    assert result.steps[-1].importances == {}
    assert result.steps[-1].eliminated_features == ()
    assert all(step.best_iteration == 2 for step in result.steps)
    assert all(
        step.fit_seconds >= 0 and step.evaluation_seconds >= 0 for step in result.steps
    )


def test_supplied_split_preserves_gap_metadata_and_weighted_metric(data):
    original = data.X.clone()
    metric_rows = []

    def metric(target, prediction, weight):
        metric_rows.append((target.copy(), weight.copy()))
        return scores(target, prediction, weight)

    result, calls = run(
        data,
        {6: 3, 3: 1, 2: 2},
        metric=metric,
        holdout=HoldoutConfig(val_size=6, gap=2),
    )
    for _, train, validation in calls:
        assert train.object_id.to_list() == list(range(16))
        assert validation.object_id.to_list() == list(range(18, 24))
        assert train.numerical_features == tuple(train.X.columns)
        for field in ("y", "time", "weight", "object_id", "business_type"):
            assert_series_equal(
                getattr(train, field), getattr(data.take(range(16)), field)
            )
            assert_series_equal(
                getattr(validation, field), getattr(data.take(range(18, 24)), field)
            )
    assert len(metric_rows) == 3
    np.testing.assert_array_equal(metric_rows[0][1], data.weight.tail(6).to_numpy())
    assert result.steps[0].metrics["secondary"] == 1
    assert_frame_equal(data.X, original)


def test_reranks_retrained_subsets_and_preserves_protected_columns(data):
    rankings = []

    def ranker(model, validation):
        rankings.append(tuple(validation.X.columns))
        return {name: float(i) for i, name in enumerate(reversed(validation.X.columns))}

    result, _ = run(
        data,
        {6: 3, 3: 2, 2: 1},
        settings(protected_features=("f5",)),
        ranker=ranker,
    )
    assert rankings == [("f0", "f1", "f2", "f3", "f4", "f5"), ("f0", "f1", "f5")]
    assert result.features == ("f0", "f5")
    assert all("f5" in step.features for step in result.steps)


@pytest.mark.parametrize("budget", [1, 2, 3, 6, 20])
def test_fit_budget_and_search_floor(data, budget):
    result, calls = run(
        data, {size: 1 for size in range(1, 7)}, settings(max_fits=budget)
    )
    assert len(calls) <= budget
    assert len(result.steps[0].features) == 6
    counts = [len(step.features) for step in result.steps]
    assert all(left > right for left, right in zip(counts, counts[1:]))
    assert counts[-1] == (6 if budget == 1 else 2)


def test_baseline_only_still_evaluates_without_ranking(data):
    def fail(*args):
        pytest.fail("Baseline-only evaluation must not request importances")

    result, calls = run(data, {6: 1}, settings(min_features=6), ranker=fail)
    assert result.features == tuple(data.X.columns)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "values",
    [
        {"f0": 1},
        {f"f{i}": 1 for i in range(7)},
        {f"f{i}": float("nan") for i in range(6)},
    ],
)
def test_invalid_ranker_results_fail(data, values):
    with pytest.raises(ValueError, match="importance"):
        run(data, {6: 1}, ranker=lambda model, data: values)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_objective_fails(data, value):
    with pytest.raises(ValueError, match="must be finite"):
        run(data, {6: 1}, metric=lambda y, p, w: MetricScores({"objective": value}))


def test_missing_objective_fails_with_available_names(data):
    with pytest.raises(ValueError, match="Available metrics.*other"):
        run(data, {6: 1}, metric=lambda y, p, w: MetricScores({"other": 1.0}))


@pytest.mark.parametrize(
    "overrides, error",
    [
        ({"min_features": 0}, ValueError),
        ({"metric": ""}, ValueError),
        ({"metric": " "}, ValueError),
        ({"max_fits": 0}, ValueError),
        ({"score_tolerance": -1}, ValueError),
        ({"score_tolerance": float("nan")}, ValueError),
        ({"protected_features": ("f0", "f0")}, ValueError),
        ({"protected_features": ("",)}, ValueError),
        ({"protected_features": ("f0", "f1", "f2")}, ValueError),
    ],
)
def test_invalid_settings(overrides, error):
    with pytest.raises(error):
        settings(**overrides)


@pytest.mark.parametrize(
    "config",
    [
        settings(min_features=7),
        settings(protected_features=("missing",)),
    ],
)
def test_invalid_data_configuration_fails_before_fit(data, config):
    def fail(params):
        pytest.fail("Invalid selection must not train")

    with pytest.raises(ValueError):
        partition = split_holdout(data, HoldoutConfig(val_size=0.25))
        replace(config, ranker=ordered_ranker).select(
            train=partition.train,
            validation=partition.val,
            params={},
            model_factory=fail,
            metric=scores,
        )


def test_standalone_selection_does_not_log_or_call_native_backend(data, monkeypatch):
    import mlflow

    def fail(*args, **kwargs):
        pytest.fail("Standalone selection must use only adapter callbacks")

    monkeypatch.setattr(mlflow, "start_run", fail)
    monkeypatch.setattr(mlflow, "log_dict", fail)
    monkeypatch.setattr(catboost.CatBoost, "select_features", fail)
    result, _ = run(data, {6: 3, 3: 1, 2: 2})
    assert result.selected_step == 1


def test_selector_requires_ranker_at_construction():
    with pytest.raises(TypeError, match="ranker"):
        RFESelector(min_features=1, metric="mae", direction="minimize")


def test_importance_ranker_uses_explicit_adapter_contract(data):
    expected = {name: float(index) for index, name in enumerate(data.X.columns)}

    class Model:
        def feature_importances(self, rows):
            assert rows is data
            return expected

    assert importance_ranker(Model(), data) is expected


@pytest.mark.parametrize(
    "leaves, constant",
    [
        ([3.0, 3.0, 5.0], True),
        ([3.0, 6.0, 3.0, 6.0, 5.0, 7.0], True),
        ([3.0, 4.0, 5.0], False),
        ([3.0, float("nan"), 5.0], False),
    ],
)
def test_nonfinite_catboost_importance_is_zero_only_for_constant_trees(
    data, leaves, constant
):
    class Native(catboost.CatBoost):
        @property
        def feature_names_(self):
            return data.X.columns

        def get_feature_importance(self, **kwargs):
            return np.array([0.0] + [float("nan")] * 5)

        def get_tree_leaf_counts(self):
            return np.array([2, 1])

        def get_leaf_values(self):
            return np.asarray(leaves)

    from ml_toolkit.models._catboost import feature_importances

    model = SimpleNamespace(
        feature_importances=lambda rows: feature_importances(Native(), rows.X.columns)
    )
    if constant:
        assert importance_ranker(model, data) == dict.fromkeys(data.X.columns, 0.0)
    else:
        with pytest.raises(ValueError, match="importances must be finite"):
            importance_ranker(model, data)


def test_shap_ranker_uses_aligned_weights_and_fixed_positive_sample(data, monkeypatch):
    import ml_toolkit.training.rankers as rankers

    data = replace(data, weight=pl.Series("weight", [0.0] * 4 + list(range(1, 21))))
    sampled = []

    def contributions(model, frame, cat_features):
        sampled.append(frame["f0"].to_numpy())
        values = np.column_stack(
            [frame[name].to_numpy() for name in frame.columns] + [np.zeros(len(frame))]
        )
        return frame.columns, frame.to_pandas(), values

    monkeypatch.setattr(rankers, "shap_contributions", contributions)
    ranker = ShapRanker(max_samples=5, seed=42)
    model = SimpleNamespace(native_model=object())
    first = ranker(model, data)
    second = ranker(model, data.select_features(["f0", "f2"]))
    np.testing.assert_array_equal(sampled[0], sampled[1])
    assert len(sampled[0]) == 5 and (sampled[0] >= 4).all()
    index = sampled[0].astype(int)
    expected = np.average(index, weights=data.weight.to_numpy()[index])
    assert first["f0"] == second["f0"] == pytest.approx(expected)


def test_real_shap_ranker_handles_null_and_numeric_categories_without_shap_extra(
    monkeypatch,
):
    import builtins

    original_import = builtins.__import__

    def without_shap(name, *args, **kwargs):
        if name == "shap" or name.startswith("shap."):
            raise ImportError("SHAP extra is unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_shap)
    rng = np.random.default_rng(42)
    count = 32
    data = Pool(
        X=pl.DataFrame({"x": rng.normal(size=count), "category": [None, 1, 2, 1] * 8}),
        y=pl.Series("target", rng.uniform(1, 3, count)),
        time=pl.Series(
            "date", [date(2024, 1, 1) + timedelta(days=i) for i in range(count)]
        ),
        weight=pl.Series("weight", rng.uniform(0.5, 2, count)),
        cat_features=("category",),
    )
    model = CatBoostRegressionModel(
        {
            "iterations": 5,
            "depth": 2,
            "thread_count": 1,
            "verbose": False,
            "allow_writing_files": False,
        }
    ).fit(data)
    values = model.native_model.get_feature_importance(
        data.to_catboost(), type="ShapValues"
    )
    expected = np.average(abs(values[:, :-1]), weights=data.weight.to_numpy(), axis=0)
    result = ShapRanker(max_samples=count)(model, data)
    np.testing.assert_allclose(list(result.values()), expected)
    partition = split_holdout(data, HoldoutConfig(val_size=0.25))
    selection = RFESelector(
        ranker=ShapRanker(max_samples=8),
        min_features=1,
        metric="mae",
        direction="minimize",
    ).select(
        train=partition.train,
        validation=partition.val,
        params=model.params,
        model_factory=CatBoostRegressionModel,
        metric=lambda y, p, w: MetricScores({"mae": np.average(abs(y - p), weights=w)}),
    )
    assert len(selection.steps) == 2
    assert all(
        np.isfinite(list(step.importances.values())).all() for step in selection.steps
    )


def test_selection_requires_matching_columns_and_nonempty_supplied_windows(data):
    def fail(params):
        pytest.fail("Invalid split must fail before any fit")

    kwargs = dict(
        params={},
        model_factory=fail,
        metric=scores,
    )
    with pytest.raises(ValueError, match="features must match"):
        replace(settings(), ranker=ordered_ranker).select(
            train=data.take(range(18)),
            validation=data.take(range(18, 24)).select_features(["f0"]),
            **kwargs,
        )
    with pytest.raises(ValueError, match="empty"):
        replace(settings(), ranker=ordered_ranker).select(
            train=data.take([]), validation=data, **kwargs
        )
