"""The reporting contract across fits, stages and artifact boundaries."""

from ml_toolkit.training import Holdout, TimeSeriesCV

from dataclasses import replace
from pathlib import Path
import json

import mlflow
import numpy as np
import polars as pl
import pytest

from ml_toolkit.data import Pool
from ml_toolkit.config import (
    ReportingConfig,
    ShapConfig,
    TrackingConfig,
    TrainingConfig,
    TuningConfig,
)
from ml_toolkit.evaluation import PredictionEvaluator, Calibration, comparison_pair
from ml_toolkit.feature_selection import RFESelector
from ml_toolkit.training import train_model, importance_ranker
from ml_toolkit.training.reporting import report_predictions, upload_artifacts
from tests.support.training import (
    MeanModel,
    make_data,
    mae,
    setup_mlflow,
    list_all_artifacts,
)
from tests.support.evaluators import CallbackEvaluator, CallbackAnalyzer


@pytest.mark.parametrize("interleaved", [False, True])
def test_report_subsets_preserve_alignment_and_share_contiguous_buffers(interleaved):
    from ml_toolkit.training.evaluation import _combine_pools

    source = make_data()
    source = Pool(source.X, source.y, source.time, source.weight)
    train, heldout = source.take(range(16)), source.take(range(16, 24))
    combined = _combine_pools(train, heldout)
    predictions = np.arange(24, dtype=float)
    labels = pl.Series(["train"] * 16 + ["val"] * 8)
    if interleaved:
        order = np.array(
            [i for pair in zip(range(8), range(16, 24)) for i in pair]
            + list(range(8, 16))
        )
        combined = combined.take(order)
        predictions, labels = predictions[order], labels.gather(order)
    part, actual_train, train_pred, actual_heldout, heldout_pred = comparison_pair(
        combined, predictions, labels
    )
    assert part == "val"
    for label, rows, values in (
        ("train", actual_train, train_pred),
        ("val", actual_heldout, heldout_pred),
    ):
        indices = np.flatnonzero((labels == label).to_numpy())
        assert rows.X.equals(combined.X[indices])
        np.testing.assert_array_equal(values, predictions[indices])
        if not interleaved:
            assert np.shares_memory(rows.X["x"].to_numpy(), source.X["x"].to_numpy())
            assert np.shares_memory(values, predictions)


@pytest.mark.parametrize(
    "flags",
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, True),
    ],
)
def test_stage_flags_gate_whole_reporting_without_affecting_metrics(
    tmp_path, monkeypatch, flags
):
    setup_mlflow(tmp_path)
    called, written, predictions, prepared, builtin = [], [], [], [], []
    analyzed, analysis_written = [], []

    class Model(MeanModel):
        def predict_pool(self, data):
            predictions.append(len(data))
            return super().predict_pool(data)

    def evaluate(**inputs):
        called.append((mlflow.active_run().data.tags["mlflow.runName"], inputs))
        return {"val/extra": 3.0}

    def writer(**inputs):
        written.append(inputs)
        return []

    import ml_toolkit.training.evaluation as module

    combine = module._combine_pools

    def capture(*args):
        prepared.append(1)
        return combine(*args)

    monkeypatch.setattr(module, "_combine_pools", capture)
    for name in ("_psi_artifacts", "_shap_artifacts"):
        monkeypatch.setattr(
            "ml_toolkit.training.reporting." + name,
            lambda *args: builtin.append(1) or [],
        )
    result = train_model(
        splitter=TimeSeriesCV(make_data(), n_splits=2, test_size=4),
        params={},
        model_factory=Model,
        metric=mae,
        run_name="stages",
        data_analyzers=CallbackAnalyzer(
            name="data_audit",
            evaluate=lambda **inputs: analyzed.append(inputs)
            or {"rows": float(len(inputs["data"]))},
            write_artifacts=lambda **inputs: analysis_written.append(inputs) or [],
        ),
        prediction_evaluators=CallbackEvaluator(
            name="business", evaluate=evaluate, write_artifacts=writer
        ),
        param_space=lambda trial: {},
        config=TrainingConfig(
            tuning=TuningConfig(metric="mae", direction="minimize", n_trials=1),
            reporting=ReportingConfig(
                on_tuning=flags[0],
                on_validation=flags[1],
                on_test=flags[2],
                on_final_fit=flags[2],
                psi=True,
                shap=ShapConfig(),
            ),
            tracking=TrackingConfig(log_test_model=False, log_final_model=False),
        ),
    )
    expected = 2 * flags[0] + 2 * flags[1] + flags[2]
    assert len(called) == len(written) == len(prepared) == expected
    assert len(analyzed) == len(analysis_written) == expected
    assert len(builtin) == expected * 2
    for (_, inputs), analysis, output in zip(called, analyzed, analysis_written):
        assert set(analysis) == {"data", "split"}
        assert analysis["data"] is inputs["data"] is output["data"]
        assert analysis["split"] is inputs["split"] is output["split"]
        assert "prediction" not in output
    assert len(predictions) == 10  # Two samples for each of five fits.
    assert result.test_metrics and "val/extra" not in result.test_metrics
    assert "val/extra" not in result.validation.mean_metrics
    for (run, inputs), output in zip(called, written):
        assert inputs["data"] is output["data"]
        assert inputs["prediction"] is output["prediction"]
        assert "stage" not in output
        parts = inputs["split"].unique(maintain_order=True).to_list()
        assert parts == ["train", "test" if run == "test" else "val"]
        if run != "test":
            assert inputs["data"].X["x"].max() < 20
    runs = mlflow.search_runs(output_format="list")
    assert all(
        "business/val/extra" in r.data.metrics
        for r in runs
        if r.data.tags["mlflow.runName"] in {x[0] for x in called}
    )


def test_combined_alignment_shared_predictions_and_no_selection_calls(tmp_path):
    setup_mlflow(tmp_path)
    source = make_data()
    source = Pool(source.X, source.y, source.time, source.weight)
    data = replace(
        source,
        object_id=pl.Series("id", [i // 2 for i in range(24)]),
        business_type=pl.Series("group", ["a", "b"] * 12),
        numerical_features=("x", "unused"),
    )
    calls, predictions = [], []

    class Model(MeanModel):
        def predict_pool(self, pool):
            predictions.append(pool.X["x"].to_numpy())
            return pool.X["x"].to_numpy().astype(float)

    def record(tag):
        def evaluate(**inputs):
            pool, prediction, split = (
                inputs["data"],
                inputs["prediction"],
                inputs["split"],
            )
            x = pool.X["x"].to_numpy()
            np.testing.assert_array_equal(prediction, x)
            np.testing.assert_array_equal(pool.object_id.to_numpy(), x // 2)
            assert pool.business_type.to_list() == [
                "a" if int(v) % 2 == 0 else "b" for v in x
            ]
            assert pool.numerical_features == ("x",)
            calls.append((tag, pool, prediction, split))
            return {}

        return CallbackEvaluator(name=tag, evaluate=evaluate)

    train_model(
        splitter=TimeSeriesCV(data, n_splits=2, gap=2, test_size=4),
        params={},
        model_factory=Model,
        metric=mae,
        run_name="alignment",
        prediction_evaluators=[record("first"), record("second")],
        feature_selector=RFESelector(
            min_features=1, metric="mae", direction="minimize", ranker=importance_ranker
        ),
        config=TrainingConfig(
            tracking=TrackingConfig(log_test_model=False, log_final_model=False)
        ),
    )
    assert len(calls) == 6
    for a, b in zip(calls[::2], calls[1::2]):
        assert a[0] == "first" and b[0] == "second"
        assert all(left is right for left, right in zip(a[1:], b[1:]))
    for _, pool, _, split in calls[:4]:
        x = pool.X["x"]
        assert x.filter(split == "val").min() - x.filter(split == "train").max() == 3
        assert x.max() < 20
    assert calls[-1][1].object_id.to_list() == data.object_id.to_list()


@pytest.mark.parametrize("name", ["", ".hidden", "a/b", "../x", "a b", "имя"])
def test_bad_names(name):
    with pytest.raises(ValueError, match="Evaluator name"):
        PredictionEvaluator(name=name)


@pytest.mark.parametrize(
    "reporting,names",
    [
        (ReportingConfig(), ["same", "SAME"]),
        (ReportingConfig(psi=True), ["PSI"]),
        (ReportingConfig(shap=ShapConfig()), ["shap"]),
    ],
)
def test_names_checked_before_fit_even_with_disabled_stages(tmp_path, reporting, names):
    setup_mlflow(tmp_path)
    with pytest.raises(ValueError, match="unique"):
        train_model(
            splitter=Holdout(make_data()),
            params={},
            model_factory=lambda p: pytest.fail("constructed"),
            metric=mae,
            run_name="names",
            prediction_evaluators=[PredictionEvaluator(name=n) for n in names],
            config=TrainingConfig(
                reporting=replace(
                    reporting, on_validation=False, on_test=False, on_final_fit=False
                )
            ),
        )
    assert mlflow.search_runs(output_format="list") == []


def test_artifact_cleanup_namespaces_and_explicit_settings(tmp_path):
    setup_mlflow(tmp_path)
    directories = []

    def writer(**inputs):
        root = inputs["artifact_dir"]
        directories.append(root)
        (root / "val").mkdir()
        (root / "val" / "score.json").write_text("{}")
        (root / "ignored.json").write_text("{}")
        return [Path("val/score.json")]

    evaluator = CallbackEvaluator(
        name="business",
        evaluate=lambda **k: {"val/rows": float(len(k["data"]))},
        write_artifacts=writer,
    )
    train_model(
        splitter=Holdout(make_data()),
        params={},
        model_factory=MeanModel,
        metric=mae,
        run_name="artifacts",
        prediction_evaluators=evaluator,
        config=TrainingConfig(
            tracking=TrackingConfig(log_test_model=False, log_final_model=False)
        ),
    )
    assert len(directories) == 2 and all(not p.exists() for p in directories)
    client = mlflow.MlflowClient()
    for run in mlflow.search_runs(output_format="list"):
        files = list_all_artifacts(client, run.info.run_id)
        if run.data.tags["mlflow.runName"] in ("validation/split_1", "final_fit"):
            assert files == ["business/val/score.json"]
            assert "business/val/rows" in run.data.metrics
        if run.data.tags["mlflow.runName"] == "artifacts":
            path = mlflow.artifacts.download_artifacts(
                run_id=run.info.run_id, artifact_path="configuration/evaluators.json"
            )
            assert json.loads(Path(path).read_text())[0]["parameters"] == {}
    assert Calibration(n_bins=3).parameters() == {"n_bins": 3, "histogram_bins": 30}


@pytest.mark.parametrize(
    "kind", ["outside", "traversal", "symlink", "missing", "extension"]
)
def test_artifact_boundaries_are_checked_before_any_upload(tmp_path, monkeypatch, kind):
    root = tmp_path / "root"
    root.mkdir()
    good = root / "good.json"
    good.write_text("{}")
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    bad = root / "bad.exe"
    bad.write_text("bad")
    if kind == "outside":
        bad = outside
    if kind == "traversal":
        bad = Path("../outside.json")
    if kind == "missing":
        bad = Path("missing.json")
    if kind == "symlink":
        bad = root / "link.json"
        bad.symlink_to(outside)
    monkeypatch.setattr(
        "ml_toolkit.training.reporting.tracking.log_artifact",
        lambda *a, **k: pytest.fail("uploaded before validation"),
    )
    with pytest.raises(ValueError):
        upload_artifacts(root, [good, bad], "business")


@pytest.mark.parametrize(
    "values",
    [
        {"bad//key": 1.0},
        {"../bad": 1.0},
        {"metric": float("nan")},
        {"metric": float("inf")},
    ],
)
def test_metric_validation_and_exception_context(tmp_path, values):
    data = make_data()
    with pytest.raises(ValueError) as caught:
        report_predictions(
            model=MeanModel({}),
            data=data,
            prediction=np.ones(len(data)),
            split=pl.Series(["train"] * len(data)),
            evaluators=(
                CallbackEvaluator(name="business", evaluate=lambda **k: values),
            ),
            config=ReportingConfig(),
            stage="validation/split_1",
        )
    assert "evaluator=business, stage=validation/split_1" in caught.value.__notes__[0]


def test_failure_stops_following_evaluators_and_cleans_files(tmp_path):
    roots = []
    error = RuntimeError("failed")

    def writer(**inputs):
        roots.append(inputs["artifact_dir"])
        raise error

    data = make_data()
    with pytest.raises(RuntimeError) as caught:
        report_predictions(
            model=MeanModel({}),
            data=data,
            prediction=np.ones(len(data)),
            split=pl.Series(["train"] * len(data)),
            evaluators=(
                CallbackEvaluator(
                    name="first", evaluate=lambda **k: {}, write_artifacts=writer
                ),
                CallbackEvaluator(
                    name="second", evaluate=lambda **k: pytest.fail("second called")
                ),
            ),
            config=ReportingConfig(),
            stage="final_fit",
        )
    assert caught.value is error and all(not p.exists() for p in roots)


@pytest.mark.parametrize(
    "split",
    [
        pl.Series(["val"] * 23),
        pl.Series([None] * 24, dtype=pl.String),
        pl.Series(["other"] * 24),
        pl.Series([1] * 24),
        pl.Series(["train"] * 8 + ["val"] * 8 + ["test"] * 8),
    ],
)
def test_invalid_split_rejected_before_custom_code(split):
    data = make_data()
    with pytest.raises(ValueError, match="split"):
        report_predictions(
            model=MeanModel({}),
            data=data,
            prediction=np.ones(len(data)),
            split=split,
            evaluators=(
                CallbackEvaluator(
                    name="never", evaluate=lambda **k: pytest.fail("called")
                ),
            ),
            config=ReportingConfig(psi=True, shap=ShapConfig()),
            stage="final_fit",
        )
