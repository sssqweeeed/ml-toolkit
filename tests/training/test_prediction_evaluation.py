from dataclasses import dataclass, replace
import json
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
import pytest
from tests.support.evaluators import CallbackEvaluator
from tests.support.training import MeanModel, ProjectData, mae, make_data, setup_mlflow

from examples.project_pool import ProjectPool as Pool
from ml_toolkit.feature_selection import RFESelector
from ml_toolkit.config import (
    HoldoutConfig,
    PredictionEvaluationConfig,
    RegressionDiagnosticsConfig,
    ReportingConfig,
    TimeSeriesCVConfig,
    TrackingConfig,
    TrainingConfig,
    TuningConfig,
)
from ml_toolkit.data import validate_training_data
from ml_toolkit.training import (
    DatasetSplit,
    PredictionEvaluator,
    importance_ranker,
    train_model,
)
from ml_toolkit.training.reporting import report_custom_evaluation


@dataclass(frozen=True)
class IdentifiedProjectData(ProjectData):
    object_id: pl.Series | None = None

    def take(self, indices):
        index = np.asarray(indices, dtype=np.int64)
        return replace(
            self,
            X=self.X[index],
            y=self.y.gather(index),
            time=self.time.gather(index),
            weight=self.weight.gather(index),
            object_id=self.object_id.gather(index),
        )

    def select_features(self, names):
        return replace(
            self,
            X=self.X.select(names),
            cat_features=tuple(name for name in self.cat_features if name in names),
        )


@pytest.fixture
def identified_data():
    source = make_data()
    return Pool(
        source.X,
        source.y,
        source.time,
        source.weight,
        object_id=pl.Series("id", np.arange(len(source)) + 1001),
    )


class RecordingEvaluator(PredictionEvaluator):
    def __init__(self, *, name):
        super().__init__(name=name)
        self.calls = []

    def evaluate(self, *, object_id, y_pred, y_true, split):
        assert isinstance(split, DatasetSplit)
        np.testing.assert_array_equal(object_id.to_numpy(), y_true + 1000)
        np.testing.assert_allclose(y_pred, y_true - 0.75)
        self.calls.append((split, object_id.to_list(), mlflow.active_run().info.run_id))
        return {"rows": np.float64(len(object_id))}


class RecordingArtifactEvaluator(RecordingEvaluator):
    def __init__(self, *, name):
        super().__init__(name=name)
        self.artifact_calls = []

    def write_artifacts(self, *, object_id, y_pred, y_true, split, artifact_dir):
        np.testing.assert_array_equal(object_id.to_numpy(), y_true + 1000)
        np.testing.assert_allclose(y_pred, y_true - 0.75)
        assert self.calls[-1][:2] == (split, object_id.to_list())
        self.artifact_calls.append((split, artifact_dir))
        # A tiny valid SVG is enough to verify actual MLflow uploads.
        image = artifact_dir / "figures" / "summary.svg"
        image.parent.mkdir()
        image.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"/>'
        )
        (artifact_dir / "ignored.parquet").write_bytes(b"not selected for upload")
        return [image]


@pytest.mark.parametrize(
    "name", ["", " ", "business name", ".", "..", ".hidden", "../report",
             "/report", "a/b", "a\\b", "имя", "a\n", "a:", "_report", "-report"]
)
def test_evaluator_rejects_invalid_names(name):
    with pytest.raises(ValueError, match="Evaluator name"):
        RecordingEvaluator(name=name)


@pytest.mark.parametrize("name", ["business", "Quality-v2", "1_report.v3"])
def test_evaluator_name_is_required_and_read_only(name):
    class MinimalEvaluator(PredictionEvaluator):
        def evaluate(self, **inputs):
            return {}

    with pytest.raises(TypeError, match="name"):
        MinimalEvaluator()
    evaluator = MinimalEvaluator(name=name)
    assert evaluator.name == name
    with pytest.raises(AttributeError):
        evaluator.name = "other"


@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("duplicate", ["same-name", "case-only", "same-object"])
def test_duplicate_evaluators_fail_before_any_fit_or_run(tmp_path, disabled, duplicate):
    setup_mlflow(tmp_path)
    first = RecordingEvaluator(name="business")
    second = (
        first if duplicate == "same-object"
        else RecordingEvaluator(name="BUSINESS" if duplicate == "case-only" else "business")
    )
    with pytest.raises(ValueError, match="Evaluator names must be unique"):
        train_model(
            data=make_data(),
            params={},
            model_factory=lambda params: pytest.fail("Model constructed before validation"),
            metric=mae,
            run_name="duplicate-evaluators",
            prediction_evaluators=[first, second],
            config=TrainingConfig(
                validation=HoldoutConfig(),
                reporting=ReportingConfig(
                    prediction_evaluation=PredictionEvaluationConfig(stages=()) if disabled else None
                ),
            ),
        )
    assert mlflow.search_runs(output_format="list") == []


@pytest.mark.parametrize("write_artifacts", [False, True])
def test_multiple_evaluators_share_predictions_and_isolate_results(
    tmp_path, identified_data, write_artifacts
):
    setup_mlflow(tmp_path)
    calls, predictions, fits, directories = [], [], [], []

    class Model(MeanModel):
        def fit(self, train, validation=None):
            fits.append(len(train))
            return super().fit(train, validation)

        def predict_pool(self, data):
            prediction = super().predict_pool(data)
            predictions.append(prediction)
            return prediction

    class NamedEvaluator(PredictionEvaluator):
        def __init__(self, *, name, value):
            super().__init__(name=name)
            self.value = value
            self.call_count = 0

        def evaluate(self, *, object_id, y_pred, y_true, split):
            self.call_count += 1
            np.testing.assert_array_equal(object_id.to_numpy(), y_true + 1000)
            run = mlflow.active_run()
            calls.append((self.name, split, run.info.run_id, y_pred, object_id.to_list()))
            return {"mae": self.value}

        def write_artifacts(self, *, artifact_dir, **inputs):
            directories.append(artifact_dir)
            path = artifact_dir / "tables" / "predictions.csv"
            path.parent.mkdir()
            path.write_text(f"value\n{self.value}\n")
            return [path]

    evaluators = [
        NamedEvaluator(name="business", value=1.0),
        NamedEvaluator(name="quality", value=2.0),
    ]
    original = evaluators.copy()
    train_model(
        data=identified_data,
        params={},
        model_factory=Model,
        metric=mae,
        run_name="multiple-evaluators",
        prediction_evaluators=evaluators,
        param_space=lambda trial: {},
        config=TrainingConfig(
            validation=HoldoutConfig(val_size=4),
            test_size=4,
            tuning=TuningConfig(metric="mae", direction="minimize", n_trials=1),
            reporting=ReportingConfig(
                prediction_evaluation=PredictionEvaluationConfig(
                    stages=("tuning", "validation", "final"),
                    write_artifacts=write_artifacts,
                )
            ),
            tracking=TrackingConfig(log_final_model=False),
        ),
    )
    assert evaluators == original
    assert [e.call_count for e in evaluators] == [6, 6]
    assert len(fits) == 3 and len(predictions) == 6 and len(calls) == 12
    expected_ids = [
        list(range(1001, 1017)), list(range(1017, 1021)),
        list(range(1001, 1017)), list(range(1017, 1021)),
        list(range(1001, 1021)), list(range(1021, 1025)),
    ]
    for index, (first, second) in enumerate(zip(calls[::2], calls[1::2], strict=True)):
        assert (first[0], second[0]) == ("business", "quality")
        assert first[1:3] == second[1:3]
        assert first[3] is second[3] is predictions[index]
        assert first[4] == second[4] == expected_ids[index]
    client = mlflow.MlflowClient()
    for name, split, run_id, _, _ in calls:
        value = 1.0 if name == "business" else 2.0
        metrics = client.get_run(run_id).data.metrics
        assert metrics[f"{split.value}/{name}/mae"] == value
        assert f"{split.value}/custom/mae" not in metrics
        artifact = f"{name}/{split.value}/tables/predictions.csv"
        if write_artifacts:
            path = Path(client.download_artifacts(run_id, artifact))
            assert path.read_text() == f"value\n{value}\n"
        else:
            assert client.list_artifacts(run_id, name) == []
        assert client.list_artifacts(run_id, "custom") == []
    assert len(directories) == (12 if write_artifacts else 0)
    assert len(set(directories)) == len(directories)
    assert all(not path.exists() for path in directories)


def test_evaluator_failure_stops_following_evaluators_without_rollback(
    tmp_path, identified_data
):
    setup_mlflow(tmp_path)
    original = RuntimeError("evaluation failed")
    calls = []

    def first(**inputs):
        calls.append(("first", mlflow.active_run().info.run_id))
        return {"rows": float(len(inputs["object_id"]))}

    def second(**inputs):
        raise original

    with pytest.raises(RuntimeError) as raised:
        train_model(
            data=identified_data,
            params={},
            model_factory=MeanModel,
            metric=mae,
            run_name="failing-evaluator",
            prediction_evaluators=[
                CallbackEvaluator(name="first", evaluate=first),
                CallbackEvaluator(name="second", evaluate=second),
                CallbackEvaluator(
                    name="third", evaluate=lambda **kw: pytest.fail("Unexpected call")
                ),
            ],
            config=TrainingConfig(validation=HoldoutConfig(val_size=4)),
        )
    assert raised.value is original
    assert "evaluator=second, stage=validation/split_1, split=train" in original.__notes__[0]
    assert len(calls) == 1
    assert mlflow.MlflowClient().get_run(calls[0][1]).data.metrics["train/first/rows"] == 20
    assert mlflow.active_run() is None


@pytest.mark.parametrize(
    "tuning, skip, repeat_cv, expected_calls",
    [
        (False, 0, False, 5),
        (False, 1, False, 3),
        (True, 0, False, 13),
        (True, 1, True, 7),
    ],
)
@pytest.mark.parametrize("custom_container", [False, True])
def test_prediction_evaluation_lifecycle_alignment_and_mlflow(
    tmp_path,
    identified_data,
    tuning,
    skip,
    repeat_cv,
    expected_calls,
    custom_container,
):
    setup_mlflow(tmp_path)
    if custom_container:
        identified_data = IdentifiedProjectData(
            identified_data.X,
            identified_data.y,
            identified_data.time,
            identified_data.weight,
            object_id=identified_data.object_id,
        )
    predictions = []

    class Model(MeanModel):
        def predict_pool(self, data):
            predictions.append(len(data))
            return data.X["x"].to_numpy() + 0.25

    evaluator: PredictionEvaluator = RecordingArtifactEvaluator(name="business")
    result = train_model(
        data=identified_data,
        params={},
        model_factory=Model,
        metric=mae,
        run_name="prediction evaluation",
        prediction_evaluators=evaluator,
        param_space=lambda trial: {"parameter": trial.suggest_int("parameter", 1, 2)},
        config=TrainingConfig(
            validation=TimeSeriesCVConfig(n_splits=2, skip_splits=skip),
            tuning=(
                TuningConfig(metric="mae", direction="minimize", n_trials=2)
                if tuning
                else None
            ),
            reporting=ReportingConfig(
                diagnostics=(
                    RegressionDiagnosticsConfig(task="regression", plots=None)
                    if repeat_cv
                    else None
                ),
                prediction_evaluation=PredictionEvaluationConfig(
                    stages=("tuning", "validation", "final")
                ),
            ),
        ),
        feature_selector=RFESelector(
            ranker=importance_ranker, min_features=1, metric="mae", direction="minimize"
        ),
    )
    assert len(evaluator.calls) == expected_calls
    assert len(evaluator.artifact_calls) == expected_calls
    assert len({root for _, root in evaluator.artifact_calls}) == expected_calls
    assert all(not root.exists() for _, root in evaluator.artifact_calls)
    # Reports reuse predictions; each split adds exactly two RFE predictions.
    assert len(predictions) == expected_calls + 2 * (2 - skip)
    assert result.validation.mean_metrics == {"mae": 0.75}
    assert all(set(split.val_metrics) == {"mae"} for split in result.validation.splits)
    assert evaluator.calls[-1][:2] == (DatasetSplit.TRAIN, list(range(1001, 1025)))
    runs = {run.info.run_id: run for run in mlflow.search_runs(output_format="list")}
    for split, ids, run_id in evaluator.calls:
        run = runs[run_id]
        assert run.data.metrics[f"{split.value}/business/rows"] == len(ids)
        client = mlflow.MlflowClient()
        artifacts = client.list_artifacts(run_id, f"business/{split.value}/figures")
        assert [item.path for item in artifacts] == [
            f"business/{split.value}/figures/summary.svg"
        ]
        assert [
            item.path for item in client.list_artifacts(run_id, f"business/{split.value}")
        ] == [
            f"business/{split.value}/figures",
        ]
        if run.data.tags["mlflow.runName"] == "final_fit":
            assert "val/business/rows" not in run.data.metrics
    parent = next(
        run for run in runs.values() if run.data.tags["mlflow.runName"] == "prediction evaluation"
    )
    assert any("validation/mae" in run.data.metrics for run in runs.values())
    assert not any(
        key.startswith("validation/business/")
        for run in runs.values()
        for key in run.data.metrics
    )
    assert not any("business/" in key for key in parent.data.metrics)


@pytest.mark.parametrize(
    "policy,tuning,test_size",
    [
        (None, None, None),
        (PredictionEvaluationConfig(stages=("validation",), splits=("train",)), None, None),
        (PredictionEvaluationConfig(stages=("validation",), splits=("val",)), None, None),
        (PredictionEvaluationConfig(stages=("final",), splits=("train",)), None, None),
        (PredictionEvaluationConfig(stages=("final",), splits=("test",)), None, 4),
        (
            PredictionEvaluationConfig(stages=("tuning",), splits=("train",)),
            TuningConfig(metric="mae", direction="minimize", n_trials=1),
            None,
        ),
        (
            PredictionEvaluationConfig(stages=("tuning",), splits=("val",)),
            TuningConfig(metric="mae", direction="minimize", n_trials=1),
            None,
        ),
    ],
)
@pytest.mark.parametrize("as_list", [False, True])
def test_missing_ids_fail_before_any_fit_or_run(
    tmp_path, policy, tuning, test_size, as_list
):
    setup_mlflow(tmp_path)
    fitted = []

    class Model(MeanModel):
        def fit(self, train, validation=None):
            fitted.append(True)
            return super().fit(train, validation)

    kwargs = dict(
        data=make_data(),
        params={},
        model_factory=Model,
        metric=mae,
        prediction_evaluators=(
            [RecordingEvaluator(name="business"), RecordingEvaluator(name="quality")]
            if as_list else RecordingEvaluator(name="business")
        ),
        param_space=lambda trial: {},
        config=TrainingConfig(
            validation=TimeSeriesCVConfig(n_splits=2),
            test_size=test_size,
            tuning=tuning,
            reporting=ReportingConfig(prediction_evaluation=policy),
        ),
    )
    with pytest.raises(ValueError, match="object_id is required"):
        train_model(**kwargs, run_name="invalid")
    assert fitted == []
    assert mlflow.search_runs(output_format="list") == []


@pytest.mark.parametrize(
    "policy,tuning,test_size",
    [
        (PredictionEvaluationConfig(stages=()), None, None),
        (PredictionEvaluationConfig(splits=()), None, None),
        (PredictionEvaluationConfig(stages=("tuning",)), None, None),
        (PredictionEvaluationConfig(stages=("final",), splits=("test",)), None, None),
        (PredictionEvaluationConfig(stages=("validation",), splits=("test",)), None, 4),
        (PredictionEvaluationConfig(stages=("final",), splits=("val",)), None, 4),
        (
            PredictionEvaluationConfig(stages=("tuning",), splits=("test",)),
            TuningConfig(metric="mae", direction="minimize", n_trials=1),
            4,
        ),
    ],
)
@pytest.mark.parametrize("as_list", [False, True])
def test_unreachable_prediction_evaluation_calls_do_not_require_ids(
    tmp_path, policy, tuning, test_size, as_list
):
    setup_mlflow(tmp_path)
    evaluator = CallbackEvaluator(name="business",
        evaluate=lambda **kwargs: pytest.fail("Unexpected evaluator call"),
        write_artifacts=lambda **kwargs: pytest.fail("Unexpected artifact call"),
    )
    result = train_model(
        data=make_data(),
        params={},
        model_factory=MeanModel,
        metric=mae,
        run_name="unreachable-prediction-evaluation",
        param_space=lambda trial: {},
        prediction_evaluators=[evaluator] if as_list else evaluator,
        config=TrainingConfig(
            validation=HoldoutConfig(),
            test_size=test_size,
            tuning=tuning,
            reporting=ReportingConfig(prediction_evaluation=policy),
            tracking=TrackingConfig(log_final_model=False),
        ),
    )
    assert result.validation.mean_metrics
    assert (result.test_metrics is not None) == (test_size is not None)


def test_default_prediction_evaluation_uses_available_splits_and_preserves_requested_null(
    tmp_path, identified_data
):
    setup_mlflow(tmp_path)
    calls, artifacts = [], []

    def evaluate(**inputs):
        ids = inputs["object_id"].to_numpy()
        np.testing.assert_array_equal(ids, inputs["y_true"] + 1000)
        calls.append(
            (
                mlflow.active_run().data.tags["mlflow.runName"],
                inputs["split"],
                ids.tolist(),
            )
        )
        return {}

    def write_artifacts(**inputs):
        artifacts.append(inputs["split"])
        return []

    config = TrainingConfig(
        validation=HoldoutConfig(val_size=4),
        test_size=4,
        tuning=TuningConfig(metric="mae", direction="minimize", n_trials=1),
        tracking=TrackingConfig(log_final_model=False),
    )
    train_model(
        data=identified_data,
        params={},
        model_factory=MeanModel,
        metric=mae,
        run_name="default-prediction-evaluation",
        config=config,
        param_space=lambda trial: {},
        prediction_evaluators=CallbackEvaluator(name="business",
            evaluate=evaluate, write_artifacts=write_artifacts
        ),
    )
    assert calls == [
        ("validation/split_1", DatasetSplit.TRAIN, list(range(1001, 1017))),
        ("validation/split_1", DatasetSplit.VAL, list(range(1017, 1021))),
        ("final_fit", DatasetSplit.TRAIN, list(range(1001, 1021))),
        ("final_fit", DatasetSplit.TEST, list(range(1021, 1025))),
    ]
    assert artifacts == [call[1] for call in calls]
    parent = next(
        run for run in mlflow.search_runs(output_format="list")
        if run.data.tags["mlflow.runName"] == "default-prediction-evaluation"
    )
    requested = Path(
        mlflow.MlflowClient().download_artifacts(
            parent.info.run_id, "configuration/requested.json"
        )
    )
    assert json.loads(requested.read_text())["reporting"]["prediction_evaluation"] is None
    assert config.reporting.prediction_evaluation is None


@pytest.mark.parametrize(
    "result, error",
    [
        ({"valid": 1.0, "score": float("nan")}, ValueError),
        ({"score": float("inf")}, ValueError),
        ({"": 1.0}, ValueError),
        ({"bad@name": 1.0}, ValueError),
        ({"../score": 1.0}, ValueError),
        ({"/score": 1.0}, ValueError),
    ],
)
def test_invalid_results_are_rejected_before_logging(
    identified_data, monkeypatch, result, error
):
    logged = []
    monkeypatch.setattr(mlflow, "log_metrics", lambda metrics: logged.append(metrics))
    evaluator = CallbackEvaluator(name="business", evaluate=lambda **kwargs: result)
    with pytest.raises(error) as raised:
        report_custom_evaluation(
            evaluator=evaluator,
            data=identified_data,
            prediction=np.ones(24),
            split=DatasetSplit.VAL,
            name="validation/split_2",
        )
    assert logged == []
    assert "stage=validation/split_2, split=val" in raised.value.__notes__[0]


def test_empty_results_and_test_split_are_supported(identified_data, monkeypatch):
    logged = []
    splits = []
    monkeypatch.setattr(mlflow, "log_metrics", lambda metrics: logged.append(metrics))

    def evaluate(**kwargs):
        splits.append(kwargs["split"])
        return {} if kwargs["split"] == DatasetSplit.TRAIN else {"business/profit": 2.5}

    for split in (DatasetSplit.TRAIN, DatasetSplit.TEST):
        report_custom_evaluation(
            evaluator=CallbackEvaluator(name="business", evaluate=evaluate),
            data=identified_data,
            prediction=np.ones(24),
            split=split,
            name="manual",
        )
    assert splits == [DatasetSplit.TRAIN, DatasetSplit.TEST]
    assert logged == [{"test/business/business/profit": 2.5}]


@pytest.mark.parametrize("invalid", ["short", "matrix", "nonfinite"])
def test_invalid_predictions_do_not_reach_evaluator(tmp_path, identified_data, invalid):
    setup_mlflow(tmp_path)

    class Model(MeanModel):
        def predict_pool(self, data):
            if invalid == "short":
                return np.ones(len(data) - 1)
            if invalid == "matrix":
                return np.ones((len(data), 1))
            return np.full(len(data), np.nan)

    def evaluate(**kwargs):
        pytest.fail("Invalid predictions must not reach the prediction evaluator")

    with pytest.raises(ValueError, match="Predictions"):
        train_model(
            data=identified_data,
            params={},
            model_factory=Model,
            metric=mae,
            prediction_evaluators=CallbackEvaluator(name="business", evaluate=evaluate),
            run_name="invalid-predictions",
            config=TrainingConfig(validation=TimeSeriesCVConfig(n_splits=2)),
        )
    assert mlflow.active_run() is None


def test_evaluator_errors_keep_original_exception_and_context(identified_data):
    original = KeyError("missing business ID")

    def evaluate(**kwargs):
        raise original

    with pytest.raises(KeyError) as raised:
        report_custom_evaluation(
            evaluator=CallbackEvaluator(name="business", evaluate=evaluate),
            data=identified_data,
            prediction=np.ones(24),
            split=DatasetSplit.VAL,
            name="validation/split_1",
        )
    assert raised.value is original
    assert "stage=validation/split_1, split=val" in raised.value.__notes__[0]


def test_custom_container_ids_are_checked_without_requiring_pool():
    data = make_data()
    source = IdentifiedProjectData(
        data.X,
        data.y,
        data.time,
        data.weight,
        object_id=pl.Series("id", list(range(len(data)))),
    )
    validate_training_data(source)
    source = replace(source, object_id=source.object_id.head(2))
    with pytest.raises(ValueError, match="equal lengths"):
        validate_training_data(source)


@pytest.mark.parametrize(
    "policy", [None, PredictionEvaluationConfig(stages=("tuning", "validation", "final"))]
)
@pytest.mark.parametrize("evaluators", [None, []])
def test_disabled_evaluator_needs_no_ids_but_final_metrics_predict(
    tmp_path, policy, evaluators
):
    setup_mlflow(tmp_path)
    predictions = []

    class Model(MeanModel):
        def predict_pool(self, data):
            predictions.append(len(data))
            return super().predict_pool(data)

    train_model(
        data=make_data(),
        params={},
        model_factory=Model,
        metric=mae,
        run_name="no-prediction-evaluation",
        prediction_evaluators=evaluators,
        config=TrainingConfig(
            validation=TimeSeriesCVConfig(n_splits=2),
            reporting=ReportingConfig(prediction_evaluation=policy),
        ),
    )
    assert predictions == [8, 8, 16, 8, 24]


@pytest.mark.parametrize(
    "split, relative_path, absolute",
    [
        (DatasetSplit.TRAIN, "plot.PNG", False),
        (DatasetSplit.TRAIN, "plot.PNG", True),
        (DatasetSplit.VAL, "nested/plot.PNG", False),
        (DatasetSplit.VAL, "nested/plot.PNG", True),
        (DatasetSplit.TEST, "figures/plot.PNG", False),
        (DatasetSplit.TEST, "nested/report.csv", True),
        (DatasetSplit.VAL, "nested/report.parquet", False),
        (DatasetSplit.TRAIN, "nested/report.json", True),
        (DatasetSplit.TEST, "nested/report.html", False),
    ],
)
def test_artifacts_work_with_empty_metrics_and_preserve_selected_paths(
    identified_data, monkeypatch, split, relative_path, absolute
):
    uploaded = []
    roots = []

    def writer(**kwargs):
        root = kwargs["artifact_dir"]
        roots.append(root)
        image = root / relative_path
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"image")
        (root / "ignored.parquet").write_bytes(b"table")
        return [image if absolute else Path(relative_path)]

    def upload(path, artifact_path):
        assert Path(path).read_bytes() == b"image"
        uploaded.append((Path(path).name, artifact_path))

    monkeypatch.setattr(
        mlflow, "log_metrics", lambda metrics: pytest.fail("empty metrics")
    )
    monkeypatch.setattr(mlflow, "log_artifact", upload)
    report_custom_evaluation(
        evaluator=CallbackEvaluator(name="business",
            evaluate=lambda **kwargs: {}, write_artifacts=writer
        ),
        data=identified_data,
        prediction=np.ones(24),
        split=split,
        name="manual",
    )
    parent = Path(relative_path).parent
    destination = f"business/{split.value}"
    if parent != Path("."):
        destination += f"/{parent.as_posix()}"
    assert uploaded == [(Path(relative_path).name, destination)]
    assert all(not root.exists() for root in roots)


def test_artifact_skip_does_not_upload_existing_files(
    identified_data, monkeypatch
):
    roots = []

    def writer(**kwargs):
        roots.append(kwargs["artifact_dir"])
        (roots[-1] / "unused.png").write_bytes(b"image")
        return []

    monkeypatch.setattr(
        mlflow, "log_artifact", lambda *a, **k: pytest.fail("skip must not upload")
    )
    report_custom_evaluation(
        evaluator=CallbackEvaluator(name="business",
            evaluate=lambda **kwargs: {}, write_artifacts=writer
        ),
        data=identified_data,
        prediction=np.ones(24),
        split=DatasetSplit.VAL,
        name="validation/split_1",
    )
    assert all(not root.exists() for root in roots)


@pytest.mark.parametrize(
    "mode, error",
    [
        ("missing", ValueError),
        ("unsupported_extension", ValueError),
        ("outside", ValueError),
        ("symlink", ValueError),
    ],
)
def test_invalid_artifact_results_fail_before_upload(
    tmp_path,
    identified_data,
    monkeypatch,
    mode,
    error,
):
    roots = []
    external = tmp_path / "outside.png"
    external.write_bytes(b"image")

    def writer(**kwargs):
        root = kwargs["artifact_dir"]
        roots.append(root)
        image = root / "valid.png"
        image.write_bytes(b"image")
        if mode == "missing":
            return [image, root / "missing.png"]
        if mode == "unsupported_extension":
            table = root / "data.exe"
            table.write_bytes(b"table")
            return [image, table]
        if mode == "symlink":
            link = root / "link.png"
            link.symlink_to(external)
            return [link]
        return [external]

    monkeypatch.setattr(
        mlflow, "log_artifact", lambda *a, **k: pytest.fail("invalid images")
    )
    with pytest.raises(error) as raised:
        report_custom_evaluation(
            evaluator=CallbackEvaluator(name="business",
                evaluate=lambda **kwargs: {}, write_artifacts=writer
            ),
            data=identified_data,
            prediction=np.ones(24),
            split=DatasetSplit.VAL,
            name="validation/split_2",
        )
    assert "stage=validation/split_2, split=val" in raised.value.__notes__[0]
    assert all(not root.exists() for root in roots)


@pytest.mark.parametrize("source", ["writer", "upload"])
def test_artifact_errors_propagate_and_clean_temporary_directory(
    identified_data, monkeypatch, source
):
    roots = []
    original = RuntimeError("artifact failure")

    def writer(**kwargs):
        root = kwargs["artifact_dir"]
        roots.append(root)
        if source == "writer":
            raise original
        image = root / "plot.png"
        image.write_bytes(b"image")
        return [image]

    def upload(*args, **kwargs):
        raise original

    monkeypatch.setattr(mlflow, "log_artifact", upload)
    with pytest.raises(RuntimeError) as raised:
        report_custom_evaluation(
            evaluator=CallbackEvaluator(name="business",
                evaluate=lambda **kwargs: {}, write_artifacts=writer
            ),
            data=identified_data,
            prediction=np.ones(24),
            split=DatasetSplit.TRAIN,
            name="final_fit",
        )
    assert raised.value is original
    assert "stage=final_fit, split=train" in original.__notes__[0]
    assert all(not root.exists() for root in roots)


def test_internal_artifact_symlink_keeps_authors_relative_path(
    identified_data, monkeypatch
):
    uploaded = []

    def writer(**kwargs):
        root = kwargs["artifact_dir"]
        original = root / "data.csv"
        original.write_text("value\n1\n")
        nested = root / "nested"
        nested.mkdir()
        link = nested / "report.csv"
        link.symlink_to(original)
        return [link]

    monkeypatch.setattr(
        mlflow,
        "log_artifact",
        lambda path, artifact_path: uploaded.append((Path(path).name, artifact_path)),
    )
    report_custom_evaluation(
        evaluator=CallbackEvaluator(name="business",
            evaluate=lambda **kwargs: {}, write_artifacts=writer
        ),
        data=identified_data,
        prediction=np.ones(24),
        split=DatasetSplit.VAL,
        name="manual",
    )
    assert uploaded == [("report.csv", "business/val/nested")]
