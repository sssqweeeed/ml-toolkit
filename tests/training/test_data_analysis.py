"""Data reports work independently of predictions and share reporting rules."""

import json
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
import pytest

from examples.custom_data_analyzer import DataAudit
from ml_toolkit.config import (
    FigureConfig,
    ReportingConfig,
    ShapConfig,
    TrackingConfig,
    TrainingConfig,
)
from ml_toolkit.data_analysis import DataAnalyzer, NumericalFeatureDistributions
from ml_toolkit.evaluation import (
    PredictionEvaluator,
    NumericalFeatureDistributions as LegacyDistributions,
)
from ml_toolkit.training import Holdout, train_model
from ml_toolkit.training.reporting import report_data
from tests.support.evaluators import CallbackAnalyzer
from tests.support.training import (
    MeanModel,
    make_data,
    mae,
    setup_mlflow,
    list_all_artifacts,
)


def test_analyzer_only_workflow_logs_separate_settings_metrics_and_artifacts(tmp_path):
    setup_mlflow(tmp_path)
    result = train_model(
        splitter=Holdout(make_data(), test_size=4),
        params={},
        model_factory=MeanModel,
        metric=mae,
        run_name="data-only",
        data_analyzers=DataAudit(name="audit"),
        config=TrainingConfig(
            tracking=TrackingConfig(log_test_model=False, log_final_model=False)
        ),
    )
    assert set(result.validation.mean_metrics) == set(result.test_metrics) == {"mae"}
    client = mlflow.MlflowClient()
    runs = mlflow.search_runs(output_format="list")
    parent = next(run for run in runs if run.data.tags["mlflow.runName"] == "data-only")
    for file, expected in (
        ("evaluators", []),
        (
            "data_analyzers",
            [
                {
                    "name": "audit",
                    "class": "examples.custom_data_analyzer.DataAudit",
                    "parameters": {},
                }
            ],
        ),
    ):
        path = client.download_artifacts(
            parent.info.run_id, f"configuration/{file}.json"
        )
        assert json.loads(Path(path).read_text()) == expected
    reported = [run for run in runs if "audit/train/rows" in run.data.metrics]
    assert len(reported) == 2
    for run in reported:
        heldout = "test" if run.data.tags["mlflow.runName"] == "test" else "val"
        assert run.data.metrics[f"audit/{heldout}/rows"] > 0
        assert list_all_artifacts(client, run.info.run_id) == ["audit/null_counts.csv"]


@pytest.mark.parametrize(
    "analyzers,evaluators,config",
    [
        (
            [DataAnalyzer(name="same"), DataAnalyzer(name="SAME")],
            None,
            ReportingConfig(),
        ),
        (
            DataAnalyzer(name="same"),
            PredictionEvaluator(name="SAME"),
            ReportingConfig(),
        ),
        (DataAnalyzer(name="PSI"), None, ReportingConfig(psi=True)),
        (DataAnalyzer(name="SHAP"), None, ReportingConfig(shap=ShapConfig())),
    ],
)
def test_names_checked_across_both_interfaces_before_fitting(
    tmp_path, analyzers, evaluators, config
):
    setup_mlflow(tmp_path)
    with pytest.raises(ValueError, match="unique"):
        train_model(
            splitter=Holdout(make_data()),
            params={},
            metric=mae,
            model_factory=lambda params: pytest.fail("model constructed"),
            run_name="names",
            config=TrainingConfig(reporting=config),
            data_analyzers=analyzers,
            prediction_evaluators=evaluators,
        )
    assert mlflow.search_runs(output_format="list") == []


@pytest.mark.parametrize("format", ["png", "svg", "pdf"])
def test_distribution_interface_matches_legacy_plot(tmp_path, monkeypatch, format):
    data = make_data()
    calls = []

    def capture(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(
        "ml_toolkit.data_analysis.monitoring.time_feature_distribution", capture
    )
    inputs = dict(
        data=data,
        split=pl.Series(["train"] * 12 + ["test"] * 12),
        artifact_dir=tmp_path,
        figure=FigureConfig(format=format),
        features=None,
        n_bins=10,
    )
    report = NumericalFeatureDistributions(features=("x",), period="week")
    legacy = LegacyDistributions(features=("x",), period="week")
    assert report.write_artifacts(**inputs) == legacy.write_artifacts(
        prediction=np.ones(len(data)), **inputs
    )
    assert report.parameters() == legacy.parameters()
    assert calls[0][0][0].equals(calls[1][0][0])
    assert calls[0][0][1].equals(calls[1][0][1])
    assert calls[0][0][2:] == calls[1][0][2:]
    assert calls[0][1] == calls[1][1]
    assert calls[0][1]["save_fig"] == tmp_path / f"000_x.{format}"
    assert not isinstance(report, PredictionEvaluator)


@pytest.mark.parametrize(
    "split",
    [
        pl.Series(["train"] * 23),
        pl.Series([None] * 24, dtype=pl.String),
        pl.Series(["other"] * 24),
        pl.Series([1] * 24),
        pl.Series(["train"] * 8 + ["val"] * 8 + ["test"] * 8),
    ],
)
def test_data_split_validation_precedes_custom_code(split):
    with pytest.raises(ValueError, match="split"):
        report_data(
            data=make_data(),
            split=split,
            analyzers=(
                CallbackAnalyzer(
                    name="never", evaluate=lambda **inputs: pytest.fail("called")
                ),
            ),
            config=ReportingConfig(),
            stage="final",
        )


@pytest.mark.parametrize(
    "values", [{"bad//key": 1}, {"metric": float("nan")}, {"metric": float("inf")}]
)
def test_data_metrics_are_validated_with_analysis_context(values):
    with pytest.raises(ValueError) as caught:
        report_data(
            data=make_data(),
            split=pl.Series(["train"] * 24),
            analyzers=(
                CallbackAnalyzer(name="audit", evaluate=lambda **inputs: values),
            ),
            config=ReportingConfig(),
            stage="final_fit",
        )
    assert "Data analysis analyzer=audit, stage=final_fit" in caught.value.__notes__


def test_analysis_artifacts_are_selected_and_cleaned_even_on_failure(monkeypatch):
    roots, uploads = [], []

    def writer(**inputs):
        root = inputs["artifact_dir"]
        roots.append(root)
        (root / "selected.json").write_text("{}")
        (root / "ignored.json").write_text("{}")
        return [Path("selected.json")]

    monkeypatch.setattr(
        "ml_toolkit.training.reporting.tracking.log_artifact",
        lambda path, artifact_path: uploads.append((artifact_path, path.name)),
    )
    inputs = dict(
        data=make_data(),
        split=pl.Series(["train"] * 24),
        config=ReportingConfig(),
        stage="final",
    )
    report_data(
        analyzers=(
            CallbackAnalyzer(
                name="audit", evaluate=lambda **kw: {}, write_artifacts=writer
            ),
        ),
        **inputs,
    )
    assert uploads == [("audit", "selected.json")]
    assert all(not root.exists() for root in roots)

    def broken_writer(**kw):
        writer(**kw)
        raise RuntimeError("failed")

    with pytest.raises(RuntimeError, match="failed"):
        report_data(
            analyzers=(
                CallbackAnalyzer(
                    name="broken",
                    evaluate=lambda **kw: {},
                    write_artifacts=broken_writer,
                ),
            ),
            **inputs,
        )
    assert all(not root.exists() for root in roots)


def test_psi_runs_without_a_model_or_predictions(monkeypatch):
    uploads = []
    monkeypatch.setattr(
        "ml_toolkit.training.reporting.tracking.log_artifact",
        lambda path, artifact_path: uploads.append(f"{artifact_path}/{path.name}"),
    )
    report_data(
        data=make_data(),
        split=pl.Series(["train"] * 12 + ["test"] * 12),
        analyzers=(),
        config=ReportingConfig(psi=True),
        stage="standalone",
    )
    assert "psi/scores.json" in uploads and "psi/scores.png" in uploads


def test_distribution_skips_an_incomplete_pair(tmp_path):
    for labels in (["train"] * 24, ["test"] * 24):
        assert (
            NumericalFeatureDistributions().write_artifacts(
                data=make_data(),
                split=pl.Series(labels),
                artifact_dir=tmp_path,
                figure=FigureConfig(),
                features=None,
                n_bins=10,
            )
            == []
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "on_test,on_final_fit", [(False, False), (True, False), (False, True), (True, True)]
)
def test_data_analyzers_follow_separate_test_and_full_data_stages(
    tmp_path, on_test, on_final_fit
):
    setup_mlflow(tmp_path)
    calls = []

    def record(*, data, split):
        calls.append(
            (
                mlflow.active_run().data.tags["mlflow.runName"],
                len(data),
                split.unique(maintain_order=True).to_list(),
            )
        )
        return {"rows": float(len(data))}

    result = train_model(
        splitter=Holdout(make_data(), test_size=4),
        params={},
        model_factory=MeanModel,
        metric=mae,
        run_name="data-stages",
        data_analyzers=CallbackAnalyzer(name="audit", evaluate=record),
        config=TrainingConfig(
            fit_on_test=True,
            reporting=ReportingConfig(
                on_validation=False, on_test=on_test, on_final_fit=on_final_fit
            ),
            tracking=TrackingConfig(log_test_model=False, log_final_model=False),
        ),
    )
    expected = []
    if on_test:
        expected.append(("test", 24, ["train", "test"]))
    if on_final_fit:
        expected.append(("final_fit", 24, ["train"]))
    assert calls == expected
    assert result.final_model is not None
    assert result.test_metrics == {"mae": 12.0}
