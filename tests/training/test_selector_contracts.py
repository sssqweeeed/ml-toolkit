from ml_toolkit.training import Holdout
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from examples.project_pool import ProjectPool
from examples.custom_selector import FixedFeatureSelector
from ml_toolkit import train_model
from ml_toolkit.config import TrainingConfig, TrackingConfig
from ml_toolkit.feature_selection import RFESelector, TopKSelector, ForwardSelector
from ml_toolkit.metrics import MetricScores

from tests.support.training import setup_mlflow


def pool():
    return ProjectPool(
        X=pl.DataFrame({"a": range(12), "b": range(12), "c": range(12)}),
        y=pl.Series([0.0, 1.0] * 6),
        time=pl.Series([date(2025, 1, 1) + timedelta(days=i) for i in range(12)]),
        weight=pl.Series([1.0] * 12),
        numerical_features=("a", "b", "c"),
    )


def quality_metric(y, prediction, weight):
    return MetricScores({"score": float(np.average(prediction, weights=weight))})


def ranked(model, data):
    return {name: 1.0 for name in data.X.columns}


def quality_factory(quality, calls):
    class Model:
        best_iteration = None

        def __init__(self, params):
            self.params = params
            params["local"] = True

        def fit(self, train, validation=None):
            self.names = tuple(train.X.columns)
            calls.append((self, self.names, validation))
            return self

        def predict_pool(self, data):
            return np.full(len(data), quality[self.names])

        def get_refit_params(self):
            return {"budget": 1}

    return Model


def select(selector, quality):
    data = pool()
    part = next(Holdout(data).get_validation())
    calls = []
    params = {"budget": 1}
    result = selector.select(
        train=part.train,
        validation=part.heldout,
        params=params,
        model_factory=quality_factory(quality, calls),
        metric=quality_metric,
    )
    assert params == {"budget": 1}
    assert len({id(model) for model, _, _ in calls}) == len(calls)
    assert result.steps[result.selected_step].features == result.features
    assert len(result.steps) == len(calls)
    return result, calls


def test_topk_keeps_exact_size_even_if_quality_deteriorates():
    result, calls = select(
        TopKSelector(k=2, ranker=ranked, protected_features=("c",)),
        {("a", "b", "c"): 1, ("a", "c"): 10},
    )
    assert result.features == ("a", "c")
    assert [step.metrics["score"] for step in result.steps] == [1, 10]
    assert len(calls) == 2


def test_topk_retaining_all_features_needs_one_fit():
    all_result, calls = select(TopKSelector(k=3, ranker=ranked), {("a", "b", "c"): 1})
    assert len(calls) == 1
    assert all_result.settings["k"] == 3


@pytest.mark.parametrize("direction,sign", [("minimize", 1), ("maximize", -1)])
def test_forward_full_history_improvement_and_ties(direction, sign):
    quality = {
        ("a",): 3,
        ("b",): 3,
        ("c",): 4,
        ("a", "b"): 2,
        ("a", "c"): 2,
        ("a", "b", "c"): 2,
    }
    result, _ = select(
        ForwardSelector(
            metric="score", direction=direction, max_features=3, max_fits=6
        ),
        {key: sign * value for key, value in quality.items()},
    )
    assert result.features == ("a", "b")
    assert len(result.steps) == 6
    assert result.selected_step == 3
    assert result.steps[4].features == ("a", "c")
    assert result.steps[-1].features == ("a", "b", "c")


@pytest.mark.parametrize("min_improvement,expected", [(0, ("b", "c")), (3, ("b",))])
def test_forward_preserves_protected_features_and_requires_strict_improvement(
    min_improvement, expected
):
    result, _ = select(
        ForwardSelector(
            metric="score",
            direction="minimize",
            max_features=3,
            max_fits=3,
            min_improvement=min_improvement,
            protected_features=("b",),
        ),
        {("b",): 4, ("a", "b"): 2, ("b", "c"): 1},
    )
    assert result.features == expected
    assert len(result.steps) == 3


def test_forward_rejects_budget_that_cannot_fit_initial_round():
    with pytest.raises(ValueError, match="initial complete"):
        select(
            ForwardSelector(
                metric="score", direction="minimize", max_features=2, max_fits=2
            ),
            {},
        )


def test_forward_does_not_start_incomplete_following_round():
    result, _ = select(
        ForwardSelector(
            metric="score", direction="minimize", max_features=2, max_fits=4
        ),
        {("a",): 1, ("b",): 2, ("c",): 3},
    )
    assert result.features == ("a",)
    assert len(result.steps) == 3


@pytest.mark.parametrize(
    "selector",
    [
        RFESelector(
            min_features=1, metric="score", direction="minimize", ranker=ranked
        ),
        TopKSelector(k=1, ranker=ranked),
        ForwardSelector(
            metric="score", direction="minimize", max_features=2, max_fits=5
        ),
        FixedFeatureSelector(("b",)),
    ],
)
def test_selectors_integrate_with_fresh_final_fit(selector, tmp_path):
    setup_mlflow(tmp_path)
    quality = {
        tuple(names): float(len(names))
        for names in (
            ("a",),
            ("b",),
            ("c",),
            ("a", "b"),
            ("a", "c"),
            ("b", "c"),
            ("a", "b", "c"),
        )
    }
    calls = []
    result = train_model(
        splitter=Holdout(pool()),
        params={},
        model_factory=quality_factory(quality, calls),
        metric=quality_metric,
        run_name="selection",
        config=TrainingConfig(tracking=TrackingConfig(log_final_model=False)),
        feature_selector=selector,
    )
    assert calls[-1][1] == result.features
    assert calls[-1][2] is None
    assert result.feature_selection.settings
    assert len({id(model) for model, _, _ in calls}) == len(calls)


def test_selector_history_snapshots_reused_metric_dictionary():
    data = pool()
    scores = {"score": 0.0}

    class Result:
        def to_dict(self):
            return scores

    def metric(y, prediction, weight):
        scores["score"] = float(prediction[0])
        return Result()

    result = RFESelector(
        min_features=1, max_fits=3, metric="score", direction="minimize", ranker=ranked
    ).select(
        train=data,
        validation=data,
        params={},
        model_factory=quality_factory(
            {("a", "b", "c"): 0, ("a", "b"): 10, ("a",): 20}, []
        ),
        metric=metric,
    )
    assert [step.metrics["score"] for step in result.steps] == [0, 10, 20]
    assert result.features == ("a", "b", "c")
    scores["score"] = 999
    assert result.steps[-1].metrics["score"] == 20
