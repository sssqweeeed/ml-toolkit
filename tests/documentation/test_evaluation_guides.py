"""Evaluator examples preserve their documented metric, files and workflow wiring."""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from examples.custom_evaluator import PredictionAudit
from examples.render_plots import EXAMPLE, PAGES
from ml_toolkit.config import FigureConfig
from ml_toolkit.data import Pool

pytestmark = pytest.mark.documentation
ROOT = Path(__file__).resolve().parents[2]


def sample_pool():
    return Pool(
        X=pl.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0]}),
        y=pl.Series([0.0, 1.0, 0.0, 1.0]),
        time=pl.Series(
            [date(2024, 1, 1) + timedelta(days=index) for index in range(4)]
        ),
        weight=pl.Series([1.0, 2.0, 1.0, 3.0]),
    )


def test_custom_evaluator_metric_matches_saved_rows_and_skips_train_only(tmp_path):
    data = sample_pool()
    prediction = np.array([0.1, 0.8, 0.2, 0.5])
    split = pl.Series(["train", "train", "test", "test"])
    evaluator = PredictionAudit(name="audit")
    metrics = evaluator.evaluate(data=data, prediction=prediction, split=split)
    files = evaluator.write_artifacts(
        data=data,
        prediction=prediction,
        split=split,
        artifact_dir=tmp_path,
        figure=FigureConfig(),
        features=None,
        n_bins=10,
    )
    assert files == [Path("test/predictions.csv")]
    rows = pl.read_csv(tmp_path / files[0], try_parse_dates=True)
    assert rows["timestamp"].equals(data.time.tail(2))
    np.testing.assert_allclose(rows["predicted"], prediction[-2:])
    assert metrics["test/mae"] == pytest.approx(
        np.average(abs(rows["observed"] - rows["predicted"]), weights=rows["weight"])
    )
    assert metrics["test/mae"] != pytest.approx(float(abs(rows["observed"] - rows["predicted"]).mean()))
    train_only = pl.Series(["train"] * len(data))
    assert evaluator.evaluate(data=data, prediction=prediction, split=train_only) == {}
    assert (
        evaluator.write_artifacts(
            data=data,
            prediction=prediction,
            split=train_only,
            artifact_dir=tmp_path,
            figure=FigureConfig(),
            features=None,
            n_bins=10,
        )
        == []
    )
    assert evaluator.parameters() == {}


def test_pu_workflow_passes_matching_population_prior_without_training(monkeypatch):
    import ml_toolkit
    from ml_toolkit.models import CatBoostPUClassificationModel

    captured = {}
    marker = object()

    def record_call(**kwargs):
        captured.update(kwargs)
        return marker

    monkeypatch.setattr(ml_toolkit, "train_model", record_call)
    document = (ROOT / "docs/evaluation/pu.md").read_text()
    code = next(
        match["code"]
        for match in EXAMPLE.finditer(document)
        if match["name"] == "pu-workflow"
    )
    namespace = {}
    exec(compile(code, "pu-workflow", "exec"), namespace)
    data = sample_pool()
    assert (
        namespace["train_pu_reports"](data, prior=0.15, params={"iterations": 10})
        is marker
    )
    assert captured["splitter"].data is data
    assert captured["params"]["pu_prior"] == captured["metric"].prior == 0.15
    assert captured["params"]["loss"] == captured["metric"].loss == "nnPU"
    assert captured["model_factory"] is CatBoostPUClassificationModel
    assert [report.name for report in captured["prediction_evaluators"]] == [
        "pu_score_distribution",
        "pu_positive_capture",
    ]


@pytest.mark.parametrize("page", PAGES)
def test_evaluation_guides_register_independent_executable_recipes(page):
    examples = list(EXAMPLE.finditer((ROOT / page).read_text()))
    assert examples
    names = [example["name"] for example in examples]
    assert len(set(names)) == len(names)
