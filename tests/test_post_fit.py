from ml_toolkit.config import (
    RFEFeatureSelectionConfig,
    CrossValidationConfig,
    TuningConfig,
    RegressionDiagnosticsConfig,
    PostFitEvaluationConfig,
    TrainingConfig,
)

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import mlflow
import numpy as np
import polars as pl
import pytest

from ml_toolkit import Pool
from ml_toolkit.training import DatasetSplit, PostFitEvaluator, importance_ranker, train_model
from ml_toolkit.training.contracts import _validate_post_fit_evaluator
from ml_toolkit.training.cross_validation import cross_validate
from ml_toolkit.training.evaluation import _log_post_fit_evaluation
from ml_toolkit.training.tuning import tune_optuna
from test_training import MeanModel, ProjectData, mae, make_data, setup_mlflow


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


class RecordingEvaluator:
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
            cv=CrossValidationConfig(n_splits=2, skip_splits=skip),
            feature_selection=RFEFeatureSelectionConfig(min_features=1, metric_name="mae", direction="minimize"),
            diagnostics=(
                RegressionDiagnosticsConfig(
                    folds="all", task="regression", psi=None, plots=None
                )
                if repeat_cv
                else None
            ),
            tuning=(
                TuningConfig(metric="mae", direction="minimize", n_trials=2)
                if tuning
                else None
            ),
            post_fit=PostFitEvaluationConfig(stages=("tuning", "cv", "final")),
        ),
    )
    assert len(evaluator.calls) == expected_calls
    assert len(evaluator.artifact_calls) == expected_calls
    assert len({root for _, root in evaluator.artifact_calls}) == expected_calls
    assert all(not root.exists() for _, root in evaluator.artifact_calls)
    # CV reuses predictions; the two RFE fits evaluate their internal holdout.
    assert len(predictions) == expected_calls + 2
    assert result.cross_validation.mean_metrics == {"mae": 0.75}
    assert all(
        set(scores) == {"mae"} for scores in result.cross_validation.fold_metrics
    )
    assert evaluator.calls[-1][:2] == (DatasetSplit.TRAIN, list(range(1001, 1025)))
    runs = {run.info.run_id: run for run in mlflow.search_runs(output_format="list")}
    for split, ids, run_id in evaluator.calls:
        run = runs[run_id]
        assert run.data.metrics[f"{split.value}/custom/rows"] == len(ids)
        client = mlflow.MlflowClient()
        artifacts = client.list_artifacts(run_id, f"custom/{split.value}/figures")
        assert [item.path for item in artifacts] == [
            f"custom/{split.value}/figures/summary.svg"
        ]
        assert [
            item.path for item in client.list_artifacts(run_id, f"custom/{split.value}")
        ] == [
            f"custom/{split.value}/figures",
        ]
        if run.data.tags["mlflow.runName"] == "final_fit":
            assert "val/custom/rows" not in run.data.metrics
    parent = next(
        run for run in runs.values() if run.data.tags["mlflow.runName"] == "post-fit"
    )
    assert any("cv/mae" in run.data.metrics for run in runs.values())
    assert not any(
        key.startswith("cv/custom/")
        for run in runs.values()
        for key in run.data.metrics
    )
    assert not any("custom/" in key for key in parent.data.metrics)


@pytest.mark.parametrize("entry_point", ["train", "cv", "tune"])
def test_missing_ids_fail_before_any_fit_or_run(tmp_path, entry_point):
    setup_mlflow(tmp_path)
    fitted = []

    class Model(MeanModel):
        def fit(self, train, validation=None):
            fitted.append(True)
            return super().fit(train, validation)

    kwargs = dict(
        data=make_data(),
        model_factory=Model,
        metric=mae,
        post_fit_evaluator=RecordingEvaluator(),
        post_fit=PostFitEvaluationConfig(stages=("tuning", "cv", "final")),
    )
    with pytest.raises(ValueError, match="object_id is required"):
        if entry_point == "train":
            train_model(
                **{k: v for k, v in kwargs.items() if k != "post_fit"},
                params={},
                run_name="invalid",
                config=TrainingConfig(
                    cv=CrossValidationConfig(n_splits=2), post_fit=kwargs["post_fit"]
                ),
            )
        elif entry_point == "cv":
            cross_validate(
                **kwargs, params={}, config=CrossValidationConfig(n_splits=2)
            )
        else:
            tune_optuna(
                **kwargs,
                base_params={},
                param_space=lambda trial: {},
                config=TuningConfig(metric="mae", direction="minimize", n_trials=1),
                cv=CrossValidationConfig(n_splits=2),
            )
    assert fitted == []
    assert mlflow.search_runs(output_format="list") == []


@pytest.mark.parametrize(
    "result, error",
    [
        ([], TypeError),
        ({"score": 1}, TypeError),
        ({"score": True}, TypeError),
        ({"score": "1.0"}, TypeError),
        ({"score": np.array([1.0])}, TypeError),
        ({"score": float("nan")}, ValueError),
        ({"score": float("inf")}, ValueError),
        ({"": 1.0}, ValueError),
        ({"bad@name": 1.0}, ValueError),
        ({"../score": 1.0}, ValueError),
        ({"/score": 1.0}, ValueError),
        ({"only spaces": 1.0, 42: 1.0}, ValueError),
    ],
)
def test_invalid_results_are_rejected_before_logging(
    identified_data, monkeypatch, result, error
):
    logged = []
    monkeypatch.setattr(mlflow, "log_metrics", lambda metrics: logged.append(metrics))
    evaluator = SimpleNamespace(evaluate=lambda **kwargs: result)
    with pytest.raises(error) as raised:
        _log_post_fit_evaluation(
            evaluator=evaluator,
            data=identified_data,
            prediction=np.ones(24),
            split=DatasetSplit.VAL,
            name="fold_2",
        )
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
        _log_post_fit_evaluation(
            evaluator=SimpleNamespace(evaluate=evaluate),
            data=identified_data,
            prediction=np.ones(24),
            split=split,
            name="manual",
        )
    assert splits == [DatasetSplit.TRAIN, DatasetSplit.TEST]
    assert logged == [{"test/custom/business/profit": 2.5}]


@pytest.mark.parametrize("split", ["train", "validation", "other"])
def test_raw_split_strings_are_rejected(identified_data, split):
    with pytest.raises(TypeError, match="DatasetSplit"):
        _log_post_fit_evaluation(
            evaluator=SimpleNamespace(evaluate=lambda **kwargs: {}),
            data=identified_data,
            prediction=np.ones(24),
            split=split,
            name="invalid",
        )


@pytest.mark.parametrize(
    "prediction", [np.ones(23), np.ones((24, 1)), np.full(24, np.nan)]
)
def test_invalid_predictions_do_not_reach_evaluator(identified_data, prediction):
    calls = []
    evaluator = SimpleNamespace(evaluate=lambda **kwargs: calls.append(kwargs))
    with pytest.raises(ValueError, match="Predictions"):
        _log_post_fit_evaluation(
            evaluator=evaluator,
            data=identified_data,
            prediction=prediction,
            split=DatasetSplit.TRAIN,
            name="final_fit",
        )
    assert calls == []


def test_evaluator_errors_keep_original_exception_and_context(identified_data):
    original = KeyError("missing business ID")

    def evaluate(**kwargs):
        raise original

    with pytest.raises(KeyError) as raised:
        _log_post_fit_evaluation(
            evaluator=SimpleNamespace(evaluate=evaluate),
            data=identified_data,
            prediction=np.ones(24),
            split=DatasetSplit.VAL,
            name="fold_1",
        )
    assert raised.value is original
    assert "stage=fold_1, split=val" in raised.value.__notes__[0]


def test_custom_container_ids_are_checked_without_requiring_pool():
    data = make_data()
    source = IdentifiedProjectData(
        data.X,
        data.y,
        data.time,
        data.weight,
        object_id=pl.Series("id", list(range(len(data)))),
    )
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

    train_model(
        data=make_data(),
        params={},
        model_factory=Model,
        metric=mae,
        run_name="no-post-fit",
        config=TrainingConfig(
            cv=CrossValidationConfig(n_splits=2),
            post_fit=PostFitEvaluationConfig(stages=("tuning", "cv", "final")),
        ),
    )
    assert predictions == [8, 8, 16, 8]


def test_artifacts_work_with_empty_metrics_and_preserve_selected_paths(
    identified_data, monkeypatch
):
    uploaded = []
    roots = []

    def writer(**kwargs):
        root = kwargs["artifact_dir"]
        roots.append(root)
        (root / "nested").mkdir()
        (root / "nested" / "plot.PNG").write_bytes(b"image")
        (root / "ignored.parquet").write_bytes(b"table")
        return [Path("nested/plot.PNG")]

    def upload(path, artifact_path):
        assert Path(path).is_file()
        uploaded.append((Path(path).name, artifact_path))

    monkeypatch.setattr(
        mlflow, "log_metrics", lambda metrics: pytest.fail("empty metrics")
    )
    monkeypatch.setattr(mlflow, "log_artifact", upload)
    _log_post_fit_evaluation(
        evaluator=SimpleNamespace(evaluate=lambda **kwargs: {}, write_artifacts=writer),
        data=identified_data,
        prediction=np.ones(24),
        split=DatasetSplit.TEST,
        name="manual",
    )
    assert uploaded == [("plot.PNG", "custom/test/nested")]
    assert all(not root.exists() for root in roots)


@pytest.mark.parametrize("result", [None, []])
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
    _log_post_fit_evaluation(
        evaluator=SimpleNamespace(evaluate=lambda **kwargs: {}, write_artifacts=writer),
        data=identified_data,
        prediction=np.ones(24),
        split=DatasetSplit.VAL,
        name="fold_1",
    )
    assert all(not root.exists() for root in roots)


@pytest.mark.parametrize(
    "mode, error",
    [
        ("nan", TypeError),
        ("tuple", TypeError),
        ("string", TypeError),
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
        _log_post_fit_evaluation(
            evaluator=SimpleNamespace(
                evaluate=lambda **kwargs: {}, write_artifacts=writer
            ),
            data=identified_data,
            prediction=np.ones(24),
            split=DatasetSplit.VAL,
            name="fold_2",
        )
    assert "stage=fold_2, split=val" in raised.value.__notes__[0]
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
        _log_post_fit_evaluation(
            evaluator=SimpleNamespace(
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


def test_non_callable_artifact_writer_is_rejected(identified_data):
    with pytest.raises(TypeError, match="write_artifacts must be callable"):
        _validate_post_fit_evaluator(
            identified_data,
            SimpleNamespace(evaluate=lambda **kwargs: {}, write_artifacts=True),
        )
