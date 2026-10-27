"""Selection diagnostics share the existing selection namespace in each split."""

from dataclasses import replace

import mlflow
import pytest

from examples.custom_selector import FixedFeatureSelector
from ml_toolkit.config import (
    FigureConfig,
    ReportingConfig,
    TrackingConfig,
    TrainingConfig,
)
from ml_toolkit.feature_selection import ForwardSelector, RFESelector, TopKSelector
from ml_toolkit.training import TimeSeriesCV, importance_ranker, train_model
from ml_toolkit.training.reporting import report_feature_selection
from tests.plots.test_feature_selection import forward_history
from tests.support.training import (
    MeanModel,
    list_all_artifacts,
    mae,
    make_data,
    setup_mlflow,
)


@pytest.mark.parametrize(
    "selector,filename,selection_fits",
    [
        (
            RFESelector(
                min_features=1,
                metric="mae",
                direction="minimize",
                ranker=importance_ranker,
            ),
            "rfe_progress",
            2,
        ),
        (TopKSelector(k=1, ranker=importance_ranker), "top_k_ranking", 2),
        (
            ForwardSelector(
                metric="mae", direction="minimize", max_features=2, max_fits=3
            ),
            "forward_progress",
            3,
        ),
        (FixedFeatureSelector(("x",)), None, 1),
        (None, None, 0),
    ],
)
@pytest.mark.parametrize("format", ["png", "svg", "pdf"])
def test_single_run_plots_per_split_in_existing_folder_even_when_reports_disabled(
    tmp_path,
    selector,
    filename,
    selection_fits,
    format,
):
    setup_mlflow(tmp_path)
    calls = {"fit": 0, "predict": 0, "rank": 0}

    class RecordingModel(MeanModel):
        @property
        def best_iteration(self):
            return 0

        def fit(self, train, validation=None):
            calls["fit"] += 1
            return super().fit(train, validation)

        def predict_pool(self, data):
            calls["predict"] += 1
            return super().predict_pool(data)

        def feature_importances(self, data):
            calls["rank"] += 1
            return super().feature_importances(data)

    result = train_model(
        splitter=TimeSeriesCV(make_data(), n_splits=2, test_size=4),
        params={},
        model_factory=RecordingModel,
        metric=mae,
        run_name="selection-plots",
        feature_selector=selector,
        config=TrainingConfig(
            fit_on_test=True,
            reporting=ReportingConfig(
                on_validation=False, figure=FigureConfig(format=format)
            ),
            tracking=TrackingConfig(log_test_model=False, log_final_model=False),
        ),
    )
    # Two validation fits, one test fit and one final fit, plus the search fits.
    assert calls["fit"] == 2 * selection_fits + 4
    assert calls["predict"] == 2 * selection_fits + 7
    assert calls["rank"] == (2 if filename in ("rfe_progress", "top_k_ranking") else 0)
    assert len(result.validation.splits) == 2
    client = mlflow.MlflowClient()
    for run in mlflow.search_runs(output_format="list"):
        paths = list_all_artifacts(client, run.info.run_id)
        assert not any(p.startswith("feature_selection/") for p in paths)
        plots = [p for p in paths if p.endswith((".png", ".svg", ".pdf"))]
        if run.data.tags["mlflow.runName"].startswith("validation/split_"):
            expected = (
                {f"selection/{filename}.{format}", f"selection/best_iteration.{format}"}
                if filename
                else set()
            )
            if filename == "forward_progress":
                expected.add(f"selection/forward_improvement.{format}")
            assert set(plots) == expected
            assert ("selection/history.json" in paths) == (selector is not None)
            assert ("selection/n_fits" in run.data.metrics) == (selector is not None)
        else:
            assert not plots


def test_standalone_selection_does_not_render_or_log(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Standalone selection must not report")

    monkeypatch.setattr("ml_toolkit.training.reporting.report_feature_selection", fail)
    monkeypatch.setattr("ml_toolkit.plots.feature_selection.rfe_progress", fail)
    data = make_data()
    result = RFESelector(
        min_features=1,
        metric="mae",
        direction="minimize",
        ranker=importance_ranker,
    ).select(
        train=data, validation=data, params={}, model_factory=MeanModel, metric=mae
    )
    assert result.features and mlflow.active_run() is None


@pytest.mark.parametrize("failure", ["draw", "upload"])
def test_report_errors_identify_selector_and_split_and_cleanup(monkeypatch, failure):
    paths = []

    def draw(result, path, *, figure):
        paths.append(path)
        path.write_text("temporary figure")
        if failure == "draw":
            raise OSError("draw failed")

    def upload(*args, **kwargs):
        raise OSError("upload failed")

    monkeypatch.setattr("ml_toolkit.plots.feature_selection.forward_progress", draw)
    monkeypatch.setattr("ml_toolkit.training.reporting.tracking.log_artifact", upload)
    with pytest.raises(OSError, match=f"{failure} failed") as caught:
        report_feature_selection(
            forward_history(), ReportingConfig(), stage="validation/split_2"
        )
    assert "selector=forward, stage=validation/split_2" in caught.value.__notes__[0]
    assert paths and not paths[0].parent.exists()


def test_custom_selector_skips_without_creating_temporary_files(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Unknown selectors have no automatic plots")

    monkeypatch.setattr("ml_toolkit.training.reporting.TemporaryDirectory", fail)
    report_feature_selection(
        replace(forward_history(), settings={}),
        ReportingConfig(),
        stage="validation/split_1",
    )


def test_optional_diagnostics_skip_unavailable_history(monkeypatch):
    paths = []
    monkeypatch.setattr(
        "ml_toolkit.training.reporting.tracking.log_artifact",
        lambda path, artifact_path: paths.append(f"{artifact_path}/{path.name}"),
    )
    result = forward_history()
    # Only a first singleton round: no earlier model to compare or iterations.
    result = replace(result, features=("a",), selected_step=0, steps=result.steps[:3])
    report_feature_selection(result, ReportingConfig(), stage="validation/split_1")
    assert paths == ["selection/forward_progress.png"]
