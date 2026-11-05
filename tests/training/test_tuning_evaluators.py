"""Evaluator objectives share predictions without changing reporting or core scores."""

from dataclasses import replace
import json
from pathlib import Path
from threading import Lock

import mlflow
import numpy as np
import polars as pl
import pytest

from ml_toolkit.config import (
    ReportingConfig, ShapConfig, TrackingConfig, TrainingConfig, TuningConfig,
)
from ml_toolkit.evaluation import PredictionEvaluator
from ml_toolkit.training import TimeSeriesCV, train_model
from tests.support.evaluators import CallbackAnalyzer, CallbackEvaluator
from tests.support.training import MeanModel, mae, make_identified_data, setup_mlflow


class CandidateModel(MeanModel):
    def predict_pool(self, data):
        return np.full(len(data), self.params["prediction"], dtype=float)


@pytest.fixture
def inputs(tmp_path):
    setup_mlflow(tmp_path)
    data = make_identified_data()
    data = replace(data, weight=pl.Series("weight", np.arange(1, len(data) + 1, dtype=float)))
    return {
        "splitter": TimeSeriesCV(data, n_splits=2, val_size=4, test_size=4),
        "params": {"prediction": 9.0},
        "model_factory": CandidateModel,
        "metric": mae,
        "run_name": "evaluator-objective",
        "param_space": lambda trial: {"prediction": (1.0, 3.0, 2.0)[trial.number]},
    }


def configuration(evaluator=None, *, direction="minimize", reporting=None, fit_on_test=False):
    return TrainingConfig(
        tuning=TuningConfig(
            metric="mae" if evaluator is None else "val/profit",
            direction=direction, n_trials=3, evaluator=evaluator,
        ),
        reporting=reporting or ReportingConfig(
            on_tuning=False, on_validation=False, on_test=False, on_final_fit=False,
        ),
        tracking=TrackingConfig(log_test_model=False, log_final_model=False),
        fit_on_test=fit_on_test,
        verbosity="silent",
    )


@pytest.mark.parametrize("direction, winner, trial", [("minimize", 1.0, 0), ("maximize", 3.0, 1)])
@pytest.mark.parametrize("reporting_enabled", [False, True])
def test_objective_uses_aligned_rows_once_per_trial_and_stays_out_of_results(
    inputs, direction, winner, trial, reporting_enabled,
):
    parts = list(inputs["splitter"].get_validation())
    calls, predictions = [], []

    class Model(CandidateModel):
        def predict_pool(self, data):
            predictions.append(len(data))
            return super().predict_pool(data)

    def evaluate(*, data, prediction, split):
        part = parts[len(calls) // 3]
        for label, expected in (("train", part.train), ("val", part.heldout)):
            mask = split == label
            assert data.X.filter(mask).equals(expected.X)
            assert data.y.filter(mask).equals(expected.y)
            assert data.time.filter(mask).equals(expected.time)
            assert data.weight.filter(mask).equals(expected.weight)
            assert data.object_id.filter(mask).equals(expected.object_id)
            assert data.auxiliary_features.filter(mask).equals(expected.auxiliary_features)
        assert split.unique(maintain_order=True).to_list() == ["train", "val"]
        calls.append(prediction.copy())
        mask = (split == "val").to_numpy()
        return {
            "val/profit": float(np.average(prediction[mask], weights=data.weight.to_numpy()[mask])),
            "train/rows": float((split == "train").sum()),
        }

    evaluator = CallbackEvaluator(
        name="business", evaluate=evaluate,
        write_artifacts=lambda **kwargs: pytest.fail("A tuning-only evaluator wrote artifacts"),
    )
    config = configuration(
        evaluator, direction=direction, fit_on_test=True,
        reporting=ReportingConfig(
            on_tuning=reporting_enabled, on_validation=reporting_enabled,
            on_test=reporting_enabled, on_final_fit=reporting_enabled,
        ),
    )
    result = train_model(**{**inputs, "model_factory": Model}, config=config)
    assert config.tuning.evaluator is evaluator
    assert len(calls) == 6
    assert len(predictions) == 19  # Six trials, two validation fits, test, final fit.
    for part in result.validation.splits:
        assert part.tuning.best_params["prediction"] == winner
        assert part.tuning.best_score == winner
        assert part.tuning.best_trial == trial
        assert set(part.train_metrics) == set(part.val_metrics) == {"mae"}
    assert set(result.validation.mean_metrics) == set(result.test_metrics) == {"mae"}
    assert result.params["prediction"] == winner
    runs = mlflow.search_runs(output_format="list")
    trials = [r for r in runs if r.data.tags["mlflow.runName"].startswith("trial_")]
    assert len(trials) == 6
    assert all(r.data.metrics["objective"] == r.data.metrics["business/val/profit"] for r in trials)
    assert all("business/train/rows" in r.data.metrics for r in trials)
    assert all(
        not any(key.startswith("business/") for key in r.data.metrics)
        for r in runs if r not in trials
    )


@pytest.mark.parametrize("explicit_none", [False, True])
def test_ordinary_metric_objective_is_unchanged(inputs, explicit_none):
    tuning = TuningConfig(metric="mae", direction="minimize", n_trials=3)
    if explicit_none:
        tuning = replace(tuning, evaluator=None)
    config = replace(configuration(), tuning=tuning)
    result = train_model(**inputs, config=config)
    assert all(part.tuning.best_params["prediction"] == 3.0 for part in result.validation.splits)
    for part, data in zip(result.validation.splits, inputs["splitter"].get_validation(), strict=True):
        expected = np.average(abs(data.heldout.y.to_numpy() - 3.0), weights=data.heldout.weight.to_numpy())
        assert part.tuning.best_score == pytest.approx(expected)


@pytest.mark.parametrize("on_tuning", [False, True])
def test_shared_objective_reuses_scores_and_preserves_report_scheduling(inputs, on_tuning):
    evaluated, written, other, analyzed, events = [], [], [], [], []

    def evaluate(**kwargs):
        run = mlflow.active_run().data.tags["mlflow.runName"]
        evaluated.append(run)
        events.append((run, "evaluate"))
        return {"val/profit": 1.0}

    def write(**kwargs):
        run = mlflow.active_run().data.tags["mlflow.runName"]
        written.append(run)
        events.append((run, "write"))
        path = kwargs["artifact_dir"] / "scores.json"
        path.write_text("{}")
        return [path]

    evaluator = CallbackEvaluator(name="business", evaluate=evaluate, write_artifacts=write)
    reports = ReportingConfig(on_tuning=on_tuning)
    train_model(
        **inputs, config=configuration(evaluator, reporting=reports, fit_on_test=True),
        prediction_evaluators=[
            evaluator,
            CallbackEvaluator(name="other", evaluate=lambda **kwargs: other.append(1) or {}),
        ],
        data_analyzers=CallbackAnalyzer(name="audit", evaluate=lambda **kwargs: analyzed.append(1) or {}),
    )
    assert len(evaluated) == 10  # Six trial objectives and four explicitly requested reports.
    assert len(written) == len(other) == len(analyzed) == 4 + 6 * on_tuning
    for run, event in events:
        if event == "write":
            assert (run, "evaluate") in events[:events.index((run, event))]
    trials = [r for r in mlflow.search_runs(output_format="list") if r.data.tags["mlflow.runName"].startswith("trial_")]
    client = mlflow.MlflowClient()
    for trial in trials:
        assert len(client.get_metric_history(trial.info.run_id, "business/val/profit")) == 1
        assert bool(client.list_artifacts(trial.info.run_id, "business")) == on_tuning


@pytest.mark.parametrize(
    "metric, values",
    [
        ("val/profit", {}),
        ("val/profit", {"other": 1.0}),
        ("val/profit", {"val/profit": float("nan")}),
        ("val/profit", {"val/profit": float("inf")}),
        ("val/profit", {"bad//key": 1.0}),
        ("business/val/profit", {"val/profit": 1.0}),
    ],
)
def test_invalid_objective_has_evaluator_key_split_and_trial_context(inputs, metric, values):
    evaluator = CallbackEvaluator(name="business", evaluate=lambda **kwargs: values)
    config = configuration(evaluator)
    config = replace(config, tuning=replace(config.tuning, metric=metric))
    with pytest.raises(ValueError) as caught:
        train_model(**inputs, config=config)
    context = str(caught.value) + "\n" + "\n".join(caught.value.__notes__)
    assert "evaluator=business" in context
    assert f"metric={metric!r}" in context
    assert "split=1" in context
    assert "trial=0" in context


@pytest.mark.parametrize("conflict", ["prediction", "model", "analyzer", "psi", "shap"])
def test_objective_names_are_checked_before_fit(inputs, conflict):
    from ml_toolkit.model_evaluation import ModelEvaluator

    def factory(params):
        pytest.fail("Name conflict was not rejected before fitting")

    evaluator = CallbackEvaluator(name="business", evaluate=lambda **kwargs: {"val/profit": 1.0})
    overrides = {"model_factory": factory}
    reporting = ReportingConfig(on_validation=False, on_test=False, on_final_fit=False)
    if conflict == "prediction":
        overrides["prediction_evaluators"] = PredictionEvaluator(name="BUSINESS")
    elif conflict == "model":
        overrides["model_evaluators"] = ModelEvaluator(name="BUSINESS")
    elif conflict == "analyzer":
        overrides["data_analyzers"] = CallbackAnalyzer(name="BUSINESS", evaluate=lambda **kwargs: {})
    else:
        evaluator = CallbackEvaluator(name=conflict, evaluate=lambda **kwargs: {"val/profit": 1.0})
        reporting = replace(reporting, psi=conflict == "psi", shap=ShapConfig() if conflict == "shap" else None)
    with pytest.raises(ValueError, match="unique and not reserved"):
        train_model(**{**inputs, **overrides}, config=configuration(evaluator, reporting=reporting))


def test_requested_configuration_describes_evaluator_without_copying_internal_state(inputs):
    class StatefulEvaluator(CallbackEvaluator):
        def parameters(self):
            return {"scale": 2.0}

    evaluator = StatefulEvaluator(name="business", evaluate=lambda **kwargs: {"val/profit": 1.0})
    evaluator.lock = Lock()  # Neither deepcopy nor JSON encoding supports this state.
    config = configuration(evaluator)
    train_model(**inputs, config=config)
    assert config.tuning.evaluator is evaluator
    parent = next(r for r in mlflow.search_runs(output_format="list") if r.data.tags["mlflow.runName"] == inputs["run_name"])
    local = mlflow.MlflowClient().download_artifacts(parent.info.run_id, "configuration/requested.json")
    recorded = json.loads(Path(local).read_text())
    assert recorded["tuning"]["evaluator"] == {
        "name": "business",
        "class": f"{StatefulEvaluator.__module__}.{StatefulEvaluator.__qualname__}",
        "parameters": {"scale": 2.0},
    }
    assert recorded["tuning"]["metric"] == "val/profit"
    assert recorded["tracking"]["log_test_model"] is False
