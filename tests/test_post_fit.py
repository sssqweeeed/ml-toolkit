from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import mlflow
import numpy as np
import polars as pl
import pytest
from evaluators import CallbackEvaluator
from test_training import MeanModel, ProjectData, mae, make_data, setup_mlflow

from examples.project_pool import ProjectPool as Pool
from ml_toolkit.config import (
    PostFitEvaluationConfig,
    RegressionDiagnosticsConfig,
    ReportingConfig,
    RFEFeatureSelectionConfig,
    TimeSeriesCVConfig,
    TrainingConfig,
    TuningConfig,
)
from ml_toolkit.data import validate_training_data
from ml_toolkit.training import (
    DatasetSplit,
    PostFitEvaluator,
    importance_ranker,
    train_model,
)
from ml_toolkit.training.reporting import report_custom_evaluation
from ml_toolkit.training.tuning import tune_optuna


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


class RecordingEvaluator(PostFitEvaluator):
    def __init__(self):
        self.calls = []

    def evaluate(self, *, object_id, y_pred, y_true, split):
        assert isinstance(split, DatasetSplit)
        np.testing.assert_array_equal(object_id.to_numpy(), y_true + 1000)
        np.testing.assert_allclose(y_pred, y_true - 0.75)
        self.calls.append((split, object_id.to_list(), mlflow.active_run().info.run_id))
        return {"rows": np.float64(len(object_id))}


class RecordingArtifactEvaluator(RecordingEvaluator):
    def __init__(self):
        super().__init__()
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
    "tuning, skip, repeat_cv, expected_calls",
    [
        (False, 0, False, 5),
        (False, 1, False, 3),
        (True, 0, False, 13),
        (True, 1, True, 7),
    ],
)
@pytest.mark.parametrize("custom_container", [False, True])
def test_post_fit_lifecycle_alignment_and_mlflow(
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

    evaluator: PostFitEvaluator = RecordingArtifactEvaluator()
    result = train_model(
        feature_ranker=importance_ranker,
        data=identified_data,
        params={},
        model_factory=Model,
        metric=mae,
        run_name="post-fit",
        post_fit_evaluator=evaluator,
        param_space=lambda trial: {"parameter": trial.suggest_int("parameter", 1, 2)},
        config=TrainingConfig(
            validation=TimeSeriesCVConfig(n_splits=2, skip_splits=skip),
            feature_selection=RFEFeatureSelectionConfig(
                min_features=1, metric="mae", direction="minimize"
            ),
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
                post_fit=PostFitEvaluationConfig(
                    stages=("tuning", "validation", "final")
                ),
            ),
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
        assert run.data.metrics[f"{split.value}/custom/rows"] == len(ids)
        client = mlflow.MlflowClient()
        artifacts = client.list_artifacts(run_id, f"figures/custom/{split.value}/figures")
        assert [item.path for item in artifacts] == [
            f"figures/custom/{split.value}/figures/summary.svg"
        ]
        assert [
            item.path for item in client.list_artifacts(run_id, f"figures/custom/{split.value}")
        ] == [
            f"figures/custom/{split.value}/figures",
        ]
        if run.data.tags["mlflow.runName"] == "final_fit":
            assert "val/custom/rows" not in run.data.metrics
    parent = next(
        run for run in runs.values() if run.data.tags["mlflow.runName"] == "post-fit"
    )
    assert any("validation/mae" in run.data.metrics for run in runs.values())
    assert not any(
        key.startswith("validation/custom/")
        for run in runs.values()
        for key in run.data.metrics
    )
    assert not any("custom/" in key for key in parent.data.metrics)


def test_missing_ids_fail_before_any_fit_or_run(tmp_path):
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
        post_fit_evaluator=RecordingEvaluator(),
        config=TrainingConfig(validation=TimeSeriesCVConfig(n_splits=2)),
    )
    with pytest.raises(ValueError, match="object_id is required"):
        train_model(**kwargs, run_name="invalid")
    assert fitted == []
    assert mlflow.search_runs(output_format="list") == []


@pytest.mark.parametrize(
    "result, error",
    [
        ({"score": float("nan")}, ValueError),
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
    evaluator = CallbackEvaluator(evaluate=lambda **kwargs: result)
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
            evaluator=CallbackEvaluator(evaluate=evaluate),
            data=identified_data,
            prediction=np.ones(24),
            split=split,
            name="manual",
        )
    assert splits == [DatasetSplit.TRAIN, DatasetSplit.TEST]
    assert logged == [{"test/custom/business/profit": 2.5}]


@pytest.mark.parametrize(
    "prediction", [np.ones(23), np.ones((24, 1)), np.full(24, np.nan)]
)
def test_invalid_predictions_do_not_reach_evaluator(identified_data, prediction):
    from ml_toolkit.training.evaluation import calculate_metrics

    with pytest.raises(ValueError, match="Predictions"):
        calculate_metrics(mae, identified_data, prediction)


def test_evaluator_errors_keep_original_exception_and_context(identified_data):
    original = KeyError("missing business ID")

    def evaluate(**kwargs):
        raise original

    with pytest.raises(KeyError) as raised:
        report_custom_evaluation(
            evaluator=CallbackEvaluator(evaluate=evaluate),
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
    evaluator = CallbackEvaluator(evaluate=lambda **kwargs: {})
    validate_training_data(source)
    source = replace(source, object_id=source.object_id.head(2))
    with pytest.raises(ValueError, match="equal lengths"):
        validate_training_data(source)


def test_disabled_evaluator_needs_no_ids_but_final_metrics_predict(tmp_path):
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
        run_name="no-post-fit",
        config=TrainingConfig(
            validation=TimeSeriesCVConfig(n_splits=2),
            reporting=ReportingConfig(
                post_fit=PostFitEvaluationConfig(
                    stages=("tuning", "validation", "final")
                ),
            ),
        ),
    )
    assert predictions == [8, 8, 16, 8, 24]


@pytest.mark.parametrize("split", list(DatasetSplit))
@pytest.mark.parametrize("relative_path", ["plot.PNG", "nested/plot.PNG", "figures/plot.PNG"])
@pytest.mark.parametrize("absolute", [False, True])
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
        evaluator=CallbackEvaluator(
            evaluate=lambda **kwargs: {}, write_artifacts=writer
        ),
        data=identified_data,
        prediction=np.ones(24),
        split=split,
        name="manual",
    )
    parent = Path(relative_path).parent
    destination = f"figures/custom/{split.value}"
    if parent != Path("."):
        destination += f"/{parent.as_posix()}"
    assert uploaded == [("plot.PNG", destination)]
    assert all(not root.exists() for root in roots)


@pytest.mark.parametrize("result", [[]])
def test_artifact_skip_does_not_upload_existing_files(
    identified_data, monkeypatch, result
):
    roots = []

    def writer(**kwargs):
        roots.append(kwargs["artifact_dir"])
        (roots[-1] / "unused.png").write_bytes(b"image")
        return result

    monkeypatch.setattr(
        mlflow, "log_artifact", lambda *a, **k: pytest.fail("skip must not upload")
    )
    report_custom_evaluation(
        evaluator=CallbackEvaluator(
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
        ("table", ValueError),
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
        if mode == "nan":
            return float("nan")
        if mode == "tuple":
            return (image,)
        if mode == "string":
            return [str(image)]
        if mode == "missing":
            return [image, root / "missing.png"]
        if mode == "table":
            table = root / "data.parquet"
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
            evaluator=CallbackEvaluator(
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
            evaluator=CallbackEvaluator(
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
