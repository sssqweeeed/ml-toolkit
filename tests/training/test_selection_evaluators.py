"""Evaluator-driven subset choices, metadata alignment and standalone behavior."""

from dataclasses import asdict, replace
import json
from pathlib import Path
from threading import Lock

import mlflow
import numpy as np
import polars as pl
import pytest

from ml_toolkit.config import ReportingConfig, TrackingConfig, TrainingConfig
from ml_toolkit.data import Pool
from ml_toolkit.feature_selection import ForwardSelector, RFESelector
from ml_toolkit.metrics import MetricScores
from ml_toolkit.training import Holdout, TimeSeriesCV, train_model
from tests.support.evaluators import CallbackEvaluator
from tests.support.training import MeanModel, mae, make_identified_data, setup_mlflow


QUALITY = {
    ("a",): 3.0, ("b",): 1.0, ("c",): 2.0,
    ("a", "b"): 4.0, ("a", "c"): 3.5, ("b", "c"): 0.5,
    ("a", "b", "c"): 4.0,
}


class SubsetModel(MeanModel):
    def predict_pool(self, data):
        return np.full(len(data), QUALITY[tuple(data.X.columns)])


@pytest.fixture
def data():
    source = make_identified_data()
    frame = pl.DataFrame({name: np.arange(len(source), dtype=float) + index for index, name in enumerate("abc")})
    return Pool(
        frame, source.y, source.time,
        pl.Series("weight", np.arange(1, len(source) + 1, dtype=float)),
        object_id=source.object_id, auxiliary_features=source.auxiliary_features,
        numerical_features=tuple(frame.columns),
    )


def ordinary_metric(y, prediction, weight):
    return MetricScores({"score": float(np.average(prediction, weights=weight))})


def evaluator_score(*, data, prediction, split):
    mask = (split == "val").to_numpy()
    return {"score": -float(np.average(prediction[mask], weights=data.weight.to_numpy()[mask]))}


def selector(kind, evaluator=None, **options):
    defaults = {"metric": "score", "direction": "minimize", "evaluator": evaluator}
    if kind == "rfe":
        return RFESelector(
            min_features=1, max_fits=3,
            ranker=lambda model, data: dict(zip(data.X.columns, range(len(data.X.columns), 0, -1), strict=True)),
            **{**defaults, **options},
        )
    return ForwardSelector(max_features=3, max_fits=6, **{**defaults, **options})


def select(data, selection, model_factory=SubsetModel, metric=ordinary_metric):
    part = next(Holdout(data, val_size=4, test_size=4, gap=2).get_validation())
    return selection.select(
        train=part.train, validation=part.heldout, params={},
        model_factory=model_factory, metric=metric,
    )


@pytest.mark.parametrize(
    "direction, tolerance, expected",
    [("minimize", 0, ("a", "b", "c")), ("minimize", 1, ("a",)),
     ("maximize", 0, ("a", "b")), ("maximize", 1, ("a",))],
)
def test_rfe_evaluator_selects_subset_and_applies_tolerance(data, direction, tolerance, expected):
    class Model(SubsetModel):
        def predict_pool(self, pool):
            return np.full(len(pool), {3: 3.0, 2: 1.0, 1: 2.0}[len(pool.X.columns)])

    evaluator = CallbackEvaluator(name="business", evaluate=evaluator_score)
    result = select(
        data, selector("rfe", evaluator, direction=direction, score_tolerance=tolerance),
        model_factory=Model,
    )
    assert result.features == expected
    assert [step.metrics["score"] for step in result.steps] == [3.0, 1.0, 2.0]
    assert [step.objective_metrics["score"] for step in result.steps] == [-3.0, -1.0, -2.0]
    assert result.settings["metric"] == "score"
    assert result.settings["evaluator"] == "business"
    if tolerance == 0:
        ordinary = select(data, selector("rfe", direction=direction), model_factory=Model)
        assert ordinary.features != expected


@pytest.mark.parametrize(
    "direction, threshold, expected, n_fits",
    [("minimize", 0, ("a", "b"), 6), ("minimize", 1, ("a",), 5),
     ("maximize", 0, ("b", "c"), 6), ("maximize", 1, ("b",), 5)],
)
def test_forward_evaluator_guides_additions_and_strict_improvement(data, direction, threshold, expected, n_fits):
    evaluator = CallbackEvaluator(name="business", evaluate=evaluator_score)
    result = select(data, selector("forward", evaluator, direction=direction, min_improvement=threshold))
    assert result.features == expected
    assert len(result.steps) == n_fits
    assert all(step.objective_metrics["score"] == -step.metrics["score"] for step in result.steps)
    assert result.steps[result.selected_step].features == expected
    if threshold == 0:
        ordinary = select(data, selector("forward", direction=direction))
        assert ordinary.features != expected


@pytest.mark.parametrize("threshold, expected", [(0, ("a", "b")), (3, ("b",))])
def test_forward_evaluator_scores_protected_baseline(data, threshold, expected):
    calls = []

    def evaluate(**kwargs):
        calls.append(tuple(kwargs["data"].X.columns))
        return evaluator_score(**kwargs)

    evaluator = CallbackEvaluator(name="business", evaluate=evaluate)
    result = select(data, selector("forward", evaluator, protected_features=("b",), min_improvement=threshold))
    assert calls[0] == ("b",)
    assert result.features == expected
    assert len(calls) == len(result.steps)
    assert all("b" in step.features for step in result.steps)


@pytest.mark.parametrize("kind", ["rfe", "forward"])
def test_objective_preserves_alignment_and_standalone_calls_have_no_tracking(data, kind, monkeypatch):
    calls, predictions = [], []
    partition = next(Holdout(data, val_size=4, test_size=4, gap=2).get_validation())

    class Model(SubsetModel):
        def predict_pool(self, pool):
            values = pool.object_id.to_numpy().astype(float)
            predictions.append((tuple(pool.X.columns), values.copy()))
            return values

    def evaluate(*, data: Pool, prediction, split):
        for label, original in (("train", partition.train), ("val", partition.heldout)):
            mask = split == label
            assert data.X.filter(mask).equals(original.X.select(data.X.columns))
            assert data.y.filter(mask).equals(original.y)
            assert data.time.filter(mask).equals(original.time)
            assert data.weight.filter(mask).equals(original.weight)
            assert data.object_id.filter(mask).equals(original.object_id)
            assert data.auxiliary_features.filter(mask).equals(original.auxiliary_features)
            np.testing.assert_array_equal(prediction[mask.to_numpy()], original.object_id.to_numpy())
        assert data.numerical_features == tuple(data.X.columns)
        assert split.unique(maintain_order=True).to_list() == ["train", "val"]
        calls.append(tuple(data.X.columns))
        return evaluator_score(data=data, prediction=prediction, split=split)

    def fail(*args, **kwargs):
        pytest.fail("Standalone selection must not start tracking, log metrics, or write evaluator artifacts")

    for name in ("start_run", "log_metrics", "log_dict", "log_artifact"):
        monkeypatch.setattr(mlflow, name, fail)
    evaluator = CallbackEvaluator(name="business", evaluate=evaluate, write_artifacts=fail)
    result = select(data, selector(kind, evaluator), model_factory=Model)
    assert len(calls) == len(result.steps)
    assert len(predictions) == 2 * len(result.steps)
    assert data.X.columns == ["a", "b", "c"]


@pytest.mark.parametrize("kind", ["rfe", "forward"])
def test_none_preserves_ordinary_selection_and_one_prediction_per_candidate(data, kind):
    predictions = []

    class Model(SubsetModel):
        def predict_pool(self, pool):
            predictions.append(len(pool))
            return super().predict_pool(pool)

    original = selector(kind)
    implicit = select(data, original, model_factory=Model)
    assert len(predictions) == len(implicit.steps)
    explicit = select(data, replace(original, evaluator=None))
    assert implicit.features == explicit.features
    assert implicit.selected_step == explicit.selected_step
    assert implicit.settings == explicit.settings
    assert "evaluator" not in implicit.settings
    assert all(step.objective_metrics is None for step in implicit.steps)
    assert [step.metrics for step in implicit.steps] == [step.metrics for step in explicit.steps]


@pytest.mark.parametrize("kind", ["rfe", "forward"])
@pytest.mark.parametrize("values", [{}, {"other": 1.0}, {"score": float("nan")}, {"score": float("inf")}, {"bad//key": 1.0}])
def test_missing_or_invalid_evaluator_scores_do_not_fall_back_to_ordinary_metric(data, kind, values):
    evaluator = CallbackEvaluator(name="business", evaluate=lambda **kwargs: values)
    with pytest.raises(ValueError) as caught:
        select(data, selector(kind, evaluator))
    context = str(caught.value) + "\n" + "\n".join(caught.value.__notes__)
    assert "evaluator=business" in context
    assert "metric='score'" in context
    assert "features=" in context


@pytest.mark.parametrize("kind", ["rfe", "forward"])
def test_evaluator_objective_requires_exact_returned_key(data, kind):
    evaluator = CallbackEvaluator(name="business", evaluate=evaluator_score)
    with pytest.raises(ValueError, match="unavailable"):
        select(data, selector(kind, evaluator, metric="business/score"))


@pytest.mark.parametrize("kind", ["rfe", "forward"])
@pytest.mark.parametrize("invalid", ["shape", "nan", "inf"])
def test_train_predictions_are_validated_before_evaluator(data, kind, invalid):
    class Model(SubsetModel):
        def predict_pool(self, pool):
            if len(pool) == 4:
                return super().predict_pool(pool)
            if invalid == "shape":
                return np.ones((len(pool), 1))
            return np.full(len(pool), float(invalid))

    evaluator = CallbackEvaluator(name="business", evaluate=lambda **kwargs: pytest.fail("Invalid predictions reached evaluator"))
    with pytest.raises(ValueError, match="one finite value per row"):
        select(data, selector(kind, evaluator), model_factory=Model)


@pytest.mark.parametrize("kind", ["rfe", "forward"])
def test_evaluator_history_snapshots_reused_dictionary_and_preserves_undefined_core_scores(data, kind):
    scores = {}

    def evaluate(**kwargs):
        scores.update(evaluator_score(**kwargs))
        return scores

    def metric(y, prediction, weight):
        return MetricScores({"score": float(prediction[0]), "undefined": float("nan")})

    evaluator = CallbackEvaluator(name="business", evaluate=evaluate)
    result = select(data, selector(kind, evaluator), metric=metric)
    assert len({id(step.objective_metrics) for step in result.steps}) == len(result.steps)
    assert all(step.objective_metrics["score"] == -step.metrics["score"] for step in result.steps)
    assert all(np.isnan(step.metrics["undefined"]) for step in result.steps)


@pytest.mark.parametrize("kind", ["rfe", "forward"])
def test_workflow_records_objective_history_without_polluting_cv_metrics(data, kind, tmp_path):
    setup_mlflow(tmp_path)
    calls = []

    def evaluate(**kwargs):
        calls.append(tuple(kwargs["data"].X.columns))
        return evaluator_score(**kwargs)

    evaluator = CallbackEvaluator(
        name="business", evaluate=evaluate,
        write_artifacts=lambda **kwargs: pytest.fail("Selection generated evaluator artifacts"),
    )
    evaluator.lock = Lock()
    selection = selector(kind, evaluator)
    result = train_model(
        splitter=TimeSeriesCV(data, n_splits=2, val_size=4, test_size=4),
        params={}, model_factory=SubsetModel, metric=mae, run_name="selection-objective",
        config=TrainingConfig(
            reporting=ReportingConfig(on_tuning=False, on_validation=False, on_test=False, on_final_fit=False),
            tracking=TrackingConfig(log_test_model=False, log_final_model=False),
            verbosity="silent",
        ),
        feature_selector=selection,
    )
    assert selection.evaluator is evaluator
    assert len(calls) == sum(len(part.feature_selection.steps) for part in result.validation.splits)
    assert set(result.validation.mean_metrics) == set(result.test_metrics) == {"mae"}
    client = mlflow.MlflowClient()
    for run in mlflow.search_runs(output_format="list"):
        if not run.data.tags["mlflow.runName"].startswith("validation/split_"):
            continue
        local = client.download_artifacts(run.info.run_id, "selection/history.json")
        saved = json.loads(Path(local).read_text())
        chosen = saved["steps"][saved["selected_step"]]
        assert saved["settings"]["evaluator"] == "business"
        assert run.data.metrics["selection/objective_metrics/score"] == chosen["objective_metrics"]["score"]
        assert run.data.metrics["selection/metrics/mae"] == chosen["metrics"]["mae"]
    snapshot = asdict(result.feature_selection)
    assert snapshot["settings"]["evaluator"] == "business"
