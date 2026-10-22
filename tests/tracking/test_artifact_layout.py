"""Every bundled report uploads usable files in its own namespace."""

import json

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

from ml_toolkit.config import FigureConfig, ReportingConfig, ShapConfig
from ml_toolkit.evaluation import (
    Calibration,
    F1Threshold,
    Gains,
    Lift,
    NumericalFeatureDistributions,
    ObservedFitted,
    PrecisionRecall,
    PUPositiveCapture,
    PUScoreDistribution,
    TimeObservedFitted,
)
from ml_toolkit.models import CatBoostRegressionModel
from ml_toolkit.training.reporting import report_predictions
from tests.support.training import make_data


@pytest.mark.parametrize("format", ["png", "svg", "pdf"])
@pytest.mark.parametrize("heldout", ["val", "test"])
def test_all_reports_export_and_cleanup(tmp_path, monkeypatch, format, heldout):
    data = make_data().with_target(pl.Series("target", [0.0, 1.0] * 12))
    artifacts, directories = {}, set()

    def upload(path, *, artifact_path):
        assert path.is_file() and path.stat().st_size > 0
        artifacts[f"{artifact_path}/{path.name}"] = path.read_bytes()
        directories.add(
            next(p for p in path.parents if p.name.startswith("ml-toolkit-"))
        )

    monkeypatch.setattr("ml_toolkit.training.reporting.tracking.log_artifact", upload)
    config = ReportingConfig(
        psi=True, figure=FigureConfig(format=format, dpi=72), n_bins=3
    )
    evaluators = (
        PrecisionRecall(),
        F1Threshold(),
        Calibration(),
        Gains(),
        Lift(),
        PUScoreDistribution(),
        PUPositiveCapture(),
        ObservedFitted(),
        TimeObservedFitted(),
        NumericalFeatureDistributions(),
    )
    model = CatBoostRegressionModel(
        {
            "iterations": 2,
            "depth": 2,
            "thread_count": 1,
            "verbose": False,
            "allow_writing_files": False,
        }
    ).fit(data.take(range(12)))
    report_predictions(
        model=model,
        data=data,
        prediction=np.linspace(0.1, 0.9, 24),
        split=pl.Series(["train"] * 12 + [heldout] * 12),
        evaluators=evaluators,
        config=config,
        stage="final_fit",
    )
    for evaluator in evaluators:
        comparative = isinstance(
            evaluator,
            (ObservedFitted, TimeObservedFitted, NumericalFeatureDistributions),
        )
        prefix = evaluator.name + "/" + ("" if comparative else heldout + "/")
        assert any(
            key.startswith(prefix)
            and "/" not in key.removeprefix(prefix)
            and key.endswith("." + format)
            for key in artifacts
        )
        if comparative:
            assert not any(
                key.startswith(evaluator.name + "/" + part + "/")
                for key in artifacts
                for part in ("val", "test")
            )
    assert not any(key.startswith(("psi/val/", "psi/test/")) for key in artifacts)
    assert {r["feature"] for r in json.loads(artifacts["psi/scores.json"])} == {
        "x",
        "unused",
    }
    assert not plt.get_fignums()
    assert all(not folder.exists() for folder in directories)


def test_shap_artifacts_and_no_train_only_explanation(tmp_path, monkeypatch):
    pytest.importorskip("shap")
    data = make_data()
    model = CatBoostRegressionModel(
        {
            "iterations": 2,
            "depth": 2,
            "thread_count": 1,
            "verbose": False,
            "allow_writing_files": False,
        }
    ).fit(data.take(range(12)))
    artifacts = []
    monkeypatch.setattr(
        "ml_toolkit.training.reporting.tracking.log_artifact",
        lambda path, artifact_path: artifacts.append(f"{artifact_path}/{path.name}"),
    )
    config = ReportingConfig(shap=ShapConfig(max_samples=5, dependence_features=("x",)))
    report_predictions(
        model=model,
        data=data,
        prediction=model.predict_pool(data),
        split=pl.Series(["train"] * 12 + ["test"] * 12),
        evaluators=(),
        config=config,
        stage="final_fit",
    )
    assert set(artifacts) == {
        "shap/test/importance.png",
        "shap/test/beeswarm.png",
        "shap/test/dependence/000_x.png",
    }
    artifacts.clear()
    monkeypatch.setattr(
        "ml_toolkit.explainability.calculate_shap_values",
        lambda *a, **k: pytest.fail("SHAP on train only"),
    )
    report_predictions(
        model=model,
        data=data,
        prediction=model.predict_pool(data),
        split=pl.Series(["train"] * 24),
        evaluators=(),
        config=config,
        stage="final_fit",
    )
    assert artifacts == []
