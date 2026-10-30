"""Model reports obey workflow ordering, stage gates and artifact contracts."""

import json
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
import pytest

from ml_toolkit.config import (
    ReportingConfig,
    ShapConfig,
    TrackingConfig,
    TrainingConfig,
    TuningConfig,
)
from ml_toolkit.evaluation import PredictionEvaluator
from ml_toolkit.model_evaluation import ModelEvaluator, ShrinkCurve
from ml_toolkit.training import Holdout, TimeSeriesCV, train_model
from ml_toolkit.training.reporting import report_predictions
from tests.support.evaluators import CallbackAnalyzer, CallbackEvaluator
from tests.support.training import (
    MeanModel,
    PrefixModel,
    list_all_artifacts,
    make_data,
    mae,
    setup_mlflow,
)


@pytest.mark.parametrize(
    "enabled", ["none", "tuning", "validation", "test", "final_fit", "all"]
)
def test_stage_flags_order_and_task_metric_identity(tmp_path, enabled):
    setup_mlflow(tmp_path)
    calls = []
    flags = {
        f"on_{stage}": enabled in (stage, "all")
        for stage in ("tuning", "validation", "test", "final_fit")
    }

    def record(kind, inputs):
        calls.append((kind, mlflow.active_run().data.tags["mlflow.runName"], inputs))
        return {}

    class Audit(ModelEvaluator):
        def evaluate(self, **inputs):
            assert inputs["metric"] is mae
            assert isinstance(inputs["model"], MeanModel)
            return record("model", inputs) or {"extra": 3.0}

    result = train_model(
        splitter=TimeSeriesCV(make_data(), n_splits=2, test_size=4),
        params={},
        model_factory=MeanModel,
        metric=mae,
        run_name="model-stages",
        prediction_evaluators=CallbackEvaluator(
            name="prediction", evaluate=lambda **kw: record("prediction", kw)
        ),
        model_evaluators=[Audit(name="model")],
        data_analyzers=CallbackAnalyzer(
            name="data", evaluate=lambda **kw: record("data", kw)
        ),
        param_space=lambda trial: {},
        config=TrainingConfig(
            tuning=TuningConfig(metric="mae", direction="minimize", n_trials=1),
            reporting=ReportingConfig(**flags),
            fit_on_test=True,
            tracking=TrackingConfig(log_test_model=False, log_final_model=False),
        ),
    )
    expected = (
        2 * flags["on_tuning"]
        + 2 * flags["on_validation"]
        + flags["on_test"]
        + flags["on_final_fit"]
    )
    assert len(calls) == 3 * expected
    for i in range(0, len(calls), 3):
        prediction, model, data = calls[i : i + 3]
        assert [prediction[0], model[0], data[0]] == ["prediction", "model", "data"]
        assert prediction[2]["data"] is model[2]["data"] is data[2]["data"]
        assert prediction[2]["split"] is model[2]["split"] is data[2]["split"]
        assert "prediction" not in model[2]
    assert set(result.validation.mean_metrics) == {"mae"}
    assert set(result.test_metrics) == {"mae"}
    parent = next(
        run
        for run in mlflow.search_runs(output_format="list")
        if run.data.tags["mlflow.runName"] == "model-stages"
    )
    config_path = mlflow.artifacts.download_artifacts(
        run_id=parent.info.run_id, artifact_path="configuration/model_evaluators.json"
    )
    assert json.loads(Path(config_path).read_text())[0]["name"] == "model"


@pytest.mark.parametrize("other", ["prediction", "data", "model", "psi", "shap"])
def test_names_are_checked_across_all_reports_before_fitting(tmp_path, other):
    setup_mlflow(tmp_path)
    kwargs = {
        "prediction_evaluators": PredictionEvaluator(name="prediction"),
        "data_analyzers": CallbackAnalyzer(name="data", evaluate=lambda **kw: {}),
        "model_evaluators": [
            ModelEvaluator(name="model"),
            ModelEvaluator(name=other.upper()),
        ],
    }
    with pytest.raises(ValueError, match="unique"):
        train_model(
            splitter=Holdout(make_data()),
            params={},
            metric=mae,
            run_name="bad-name",
            model_factory=lambda p: pytest.fail("constructed model"),
            config=TrainingConfig(
                reporting=ReportingConfig(
                    psi=True,
                    shap=ShapConfig(),
                    on_validation=False,
                    on_test=False,
                    on_final_fit=False,
                )
            ),
            **kwargs,
        )
    assert not mlflow.search_runs(output_format="list")


def test_disabled_shrink_does_not_require_capability(tmp_path):
    setup_mlflow(tmp_path)
    train_model(
        splitter=Holdout(make_data()),
        params={},
        metric=mae,
        run_name="disabled",
        model_factory=MeanModel,
        model_evaluators=ShrinkCurve(),
        config=TrainingConfig(
            reporting=ReportingConfig(on_validation=False, on_final_fit=False),
            tracking=TrackingConfig(log_test_model=False, log_final_model=False),
        ),
    )


def test_shrink_exports_curve_without_changing_training_results(tmp_path):
    setup_mlflow(tmp_path)
    config = TrainingConfig(
        tracking=TrackingConfig(log_test_model=False, log_final_model=False)
    )
    args = dict(
        splitter=Holdout(make_data(), test_size=4),
        params={"iterations": 4},
        metric=mae,
        model_factory=PrefixModel,
        config=config,
    )
    plain = train_model(run_name="plain", **args)
    reported = train_model(
        run_name="reported", model_evaluators=ShrinkCurve(step=2), **args
    )
    assert reported.test_metrics == plain.test_metrics
    assert reported.validation.mean_metrics == plain.validation.mean_metrics
    np.testing.assert_array_equal(
        reported.model.predict_pool(make_data()), plain.model.predict_pool(make_data())
    )
    client = mlflow.MlflowClient()
    artifact_runs = []
    for run in mlflow.search_runs(output_format="list"):
        paths = list_all_artifacts(client, run.info.run_id)
        if "shrink/metrics.parquet" in paths:
            artifact_runs.append(run)
            assert "shrink/000_mae.png" in paths
    assert len(artifact_runs) == 2
    for run in artifact_runs:
        path = mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id, artifact_path="shrink/metrics.parquet"
        )
        table = pl.read_parquet(path)
        assert table["n_iterations"].unique().sort().to_list() == [1, 2, 4]
        label = "test" if run.data.tags["mlflow.runName"] == "test" else "val"
        expected = (
            reported.test_metrics
            if label == "test"
            else reported.validation.last_split.val_metrics
        )
        assert (
            table.filter((pl.col("n_iterations") == 4) & (pl.col("split") == label))[
                "value"
            ][0]
            == expected["mae"]
        )


def test_model_report_failure_cleans_directory_and_keeps_stage_iteration_context(
    tmp_path,
):
    data = make_data()
    model = PrefixModel({"iterations": 3}).fit(data)
    roots = []
    error = RuntimeError("metric failed")

    def metric(*args):
        raise error

    class Capture(ShrinkCurve):
        def write_artifacts(self, **inputs):
            roots.append(inputs["artifact_dir"])
            return super().write_artifacts(**inputs)

    with pytest.raises(RuntimeError) as caught:
        report_predictions(
            model=model,
            data=data,
            prediction=model.predict_pool(data),
            split=pl.Series(["train"] * len(data)),
            evaluators=(),
            model_evaluators=(Capture(),),
            metric=metric,
            config=ReportingConfig(),
            stage="validation/split_1",
        )
    assert caught.value is error
    assert any("n_iterations=1" in note for note in error.__notes__)
    assert any("stage=validation/split_1" in note for note in error.__notes__)
    assert roots and all(not root.exists() for root in roots)


def test_model_reports_reuse_scalar_and_artifact_validation(tmp_path, monkeypatch):
    data = make_data()
    model = MeanModel({}).fit(data)
    inputs = dict(
        model=model,
        data=data,
        prediction=model.predict_pool(data),
        split=pl.Series(["train"] * len(data)),
        evaluators=(),
        metric=mae,
        config=ReportingConfig(),
        stage="final_fit",
    )

    class BadScalar(ModelEvaluator):
        def evaluate(self, **inputs):
            return {"undefined": float("nan")}

    with pytest.raises(ValueError, match="must be finite"):
        report_predictions(**inputs, model_evaluators=(BadScalar(name="bad"),))

    class Outside(ModelEvaluator):
        def write_artifacts(self, **inputs):
            return [tmp_path / "outside.json"]

    monkeypatch.setattr(
        "ml_toolkit.training.reporting.tracking.log_artifact",
        lambda *a, **kw: pytest.fail("uploaded"),
    )
    with pytest.raises(ValueError, match="inside artifact_dir"):
        report_predictions(**inputs, model_evaluators=(Outside(name="bad"),))
