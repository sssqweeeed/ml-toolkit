from dataclasses import dataclass, replace
from types import SimpleNamespace

import mlflow
import numpy as np
import polars as pl
import pytest

from ml_toolkit import Pool
from ml_toolkit.training import DatasetSplit, PostFitEvaluator, train_model
from ml_toolkit.training.contracts import _validate_post_fit_evaluator
from ml_toolkit.training.cross_validation import cross_validate
from ml_toolkit.training.evaluation import DiagnosticsOptions, _log_post_fit_evaluation
from ml_toolkit.training.tuning import tune_optuna
from test_training import MeanModel, ProjectData, mae, make_data, setup_mlflow


@dataclass(frozen=True)
class IdentifiedProjectData(ProjectData):
    object_id: pl.Series | None = None

    def take(self, indices):
        index = np.asarray(indices, dtype=np.int64)
        return replace(self, X=self.X[index], y=self.y.gather(index),
                       time=self.time.gather(index), weight=self.weight.gather(index),
                       object_id=self.object_id.gather(index))

    def select_features(self, names):
        return replace(self, X=self.X.select(names),
                       cat_features=tuple(name for name in self.cat_features if name in names))


@pytest.fixture
def identified_data():
    source = make_data()
    return Pool(source.X, source.y, source.time, source.weight,
                object_id=pl.Series("id", np.arange(len(source)) + 1001))


class RecordingEvaluator:
    def __init__(self):
        self.calls = []

    def evaluate(self, *, object_id, y_pred, y_true, split):
        assert isinstance(split, DatasetSplit)
        np.testing.assert_array_equal(object_id.to_numpy(), y_true + 1000)
        np.testing.assert_allclose(y_pred, y_true - 0.75)
        self.calls.append((split, object_id.to_list(), mlflow.active_run().info.run_id))
        return {"rows": np.float64(len(object_id))}


@pytest.mark.parametrize("tuning, skip, repeat_cv, expected_calls", [
    (False, 0, False, 5),
    (False, 1, False, 3),
    (True, 0, False, 9),
    (True, 1, True, 7),
])
@pytest.mark.parametrize("custom_container", [False, True])
def test_post_fit_lifecycle_alignment_and_mlflow(
    tmp_path, identified_data, tuning, skip, repeat_cv, expected_calls, custom_container,
):
    setup_mlflow(tmp_path)
    if custom_container:
        identified_data = IdentifiedProjectData(
            identified_data.X, identified_data.y, identified_data.time,
            identified_data.weight, object_id=identified_data.object_id,
        )
    predictions = []

    class Model(MeanModel):
        def predict_pool(self, data):
            predictions.append(len(data))
            return data.X["x"].to_numpy() + 0.25

    evaluator: PostFitEvaluator = RecordingEvaluator()
    result = train_model(
        data=identified_data, params={}, model_factory=Model, metric=mae,
        run_name="post-fit", n_splits=2, skip_splits=skip, select_features=1,
        post_fit_evaluator=evaluator, tune_params=tuning, n_trials=2,
        param_space=lambda trial: {"parameter": trial.suggest_int("parameter", 1, 2)},
        metric_to_optimize="mae", optimize_direction="minimize",
        diagnostics=DiagnosticsOptions(task="regression", psi=False, plots=False)
        if repeat_cv else None,
    )
    assert len(evaluator.calls) == expected_calls
    # CV reuses predictions; only final fitting adds one prediction call.
    assert len(predictions) == expected_calls
    assert result.cross_validation.mean_metrics == {"mae": 0.75}
    assert all(set(scores) == {"mae"} for scores in result.cross_validation.fold_metrics)
    assert evaluator.calls[-1][:2] == (DatasetSplit.TRAIN, list(range(1001, 1025)))
    runs = {run.info.run_id: run for run in mlflow.search_runs(output_format="list")}
    for split, ids, run_id in evaluator.calls:
        run = runs[run_id]
        assert run.data.metrics[f"{split.value}/custom/rows"] == len(ids)
        if run.data.tags["mlflow.runName"] == "final_fit":
            assert "val/custom/rows" not in run.data.metrics
    parent = next(run for run in runs.values() if run.data.tags["mlflow.runName"] == "post-fit")
    assert any("cv/mae" in run.data.metrics for run in runs.values())
    assert not any(key.startswith("cv/custom/") for run in runs.values() for key in run.data.metrics)
    assert not any("custom/" in key for key in parent.data.metrics)


@pytest.mark.parametrize("entry_point", ["train", "cv", "tune"])
def test_missing_ids_fail_before_any_fit_or_run(tmp_path, entry_point):
    setup_mlflow(tmp_path)
    fitted = []

    class Model(MeanModel):
        def fit(self, train, validation=None):
            fitted.append(True)
            return super().fit(train, validation)

    kwargs = dict(data=make_data(), model_factory=Model, metric=mae,
                  post_fit_evaluator=RecordingEvaluator(), n_splits=2)
    with pytest.raises(ValueError, match="object_id is required"):
        if entry_point == "train":
            train_model(**kwargs, params={}, run_name="invalid")
        elif entry_point == "cv":
            cross_validate(**kwargs, params={})
        else:
            tune_optuna(**kwargs, base_params={}, param_space=lambda trial: {},
                        metric_to_optimize="mae", n_trials=1)
    assert fitted == []
    assert mlflow.search_runs(output_format="list") == []


@pytest.mark.parametrize("result, error", [
    ([], TypeError),
    ({"score": 1}, TypeError),
    ({"score": True}, TypeError),
    ({"score": "1.0"}, TypeError),
    ({"score": np.array([1.])}, TypeError),
    ({"score": float("nan")}, ValueError),
    ({"score": float("inf")}, ValueError),
    ({"": 1.}, ValueError),
    ({"bad@name": 1.}, ValueError),
    ({"../score": 1.}, ValueError),
    ({"/score": 1.}, ValueError),
    ({"only spaces": 1., 42: 1.}, ValueError),
])
def test_invalid_results_are_rejected_before_logging(identified_data, monkeypatch, result, error):
    logged = []
    monkeypatch.setattr(mlflow, "log_metrics", lambda metrics: logged.append(metrics))
    evaluator = SimpleNamespace(evaluate=lambda **kwargs: result)
    with pytest.raises(error) as raised:
        _log_post_fit_evaluation(evaluator=evaluator, data=identified_data,
                                 prediction=np.ones(24), split=DatasetSplit.VAL, name="fold_2")
    assert logged == []
    assert "stage=fold_2, split=val" in raised.value.__notes__[0]


def test_empty_results_and_test_split_are_supported(identified_data, monkeypatch):
    logged = []
    splits = []
    monkeypatch.setattr(mlflow, "log_metrics", lambda metrics: logged.append(metrics))

    def evaluate(**kwargs):
        splits.append(kwargs["split"])
        return {} if kwargs["split"] == DatasetSplit.TRAIN else {"business/profit": 2.5}

    for split in (DatasetSplit.TRAIN, DatasetSplit.TEST):
        _log_post_fit_evaluation(evaluator=SimpleNamespace(evaluate=evaluate),
                                 data=identified_data, prediction=np.ones(24),
                                 split=split, name="manual")
    assert splits == [DatasetSplit.TRAIN, DatasetSplit.TEST]
    assert logged == [{"test/custom/business/profit": 2.5}]


@pytest.mark.parametrize("split", ["train", "validation", "other"])
def test_raw_split_strings_are_rejected(identified_data, split):
    with pytest.raises(TypeError, match="DatasetSplit"):
        _log_post_fit_evaluation(evaluator=SimpleNamespace(evaluate=lambda **kwargs: {}),
                                 data=identified_data, prediction=np.ones(24),
                                 split=split, name="invalid")


@pytest.mark.parametrize("prediction", [np.ones(23), np.ones((24, 1)), np.full(24, np.nan)])
def test_invalid_predictions_do_not_reach_evaluator(identified_data, prediction):
    calls = []
    evaluator = SimpleNamespace(evaluate=lambda **kwargs: calls.append(kwargs))
    with pytest.raises(ValueError, match="Predictions"):
        _log_post_fit_evaluation(evaluator=evaluator, data=identified_data,
                                 prediction=prediction, split=DatasetSplit.TRAIN, name="final_fit")
    assert calls == []


def test_evaluator_errors_keep_original_exception_and_context(identified_data):
    original = KeyError("missing business ID")

    def evaluate(**kwargs):
        raise original

    with pytest.raises(KeyError) as raised:
        _log_post_fit_evaluation(evaluator=SimpleNamespace(evaluate=evaluate),
                                 data=identified_data, prediction=np.ones(24),
                                 split=DatasetSplit.VAL, name="fold_1")
    assert raised.value is original
    assert "stage=fold_1, split=val" in raised.value.__notes__[0]


def test_custom_container_ids_are_checked_without_requiring_pool():
    data = make_data()
    source = IdentifiedProjectData(data.X, data.y, data.time, data.weight,
                                   object_id=pl.Series("id", list(range(len(data)))))
    evaluator = SimpleNamespace(evaluate=lambda **kwargs: {})
    _validate_post_fit_evaluator(source, evaluator)
    source = replace(source, object_id=source.object_id.head(2))
    with pytest.raises(ValueError, match="equal lengths"):
        _validate_post_fit_evaluator(source, evaluator)
    with pytest.raises(TypeError, match="evaluate method"):
        _validate_post_fit_evaluator(source, object())


def test_disabled_evaluator_needs_no_ids_and_no_final_predictions(tmp_path):
    setup_mlflow(tmp_path)
    predictions = []

    class Model(MeanModel):
        def predict_pool(self, data):
            predictions.append(len(data))
            return super().predict_pool(data)

    train_model(data=make_data(), params={}, model_factory=Model, metric=mae,
                run_name="no-post-fit", n_splits=2)
    assert predictions == [8, 8, 16, 8]
