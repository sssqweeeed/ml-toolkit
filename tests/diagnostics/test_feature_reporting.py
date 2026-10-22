"""Shared level semantics with independent, stateless report fits."""

from dataclasses import replace
import numpy as np
import polars as pl
import pytest

from ml_toolkit.config import ReportingConfig, FigureConfig
from ml_toolkit.evaluation import (
    ObservedFitted,
    TimeObservedFitted,
    NumericalFeatureDistributions,
    Calibration,
    PrecisionRecall,
    F1Threshold,
    Gains,
    Lift,
    PUScoreDistribution,
    PUPositiveCapture,
)
from ml_toolkit.feature_levels import FeatureLevelConfig, LevelTransformer
from ml_toolkit.training.reporting import report_predictions
from tests.support.training import MeanModel, make_data


@pytest.mark.parametrize(
    "psi,observed", [(True, True), (True, False), (False, True), (False, False)]
)
def test_independent_fits_produce_identical_train_levels(monkeypatch, psi, observed):
    data = make_data()
    data = replace(
        data,
        X=data.X.with_columns(
            pl.Series("unused", ["a", "b", None] * 4 + ["unseen"] * 12)
        ),
        cat_features=("unused",),
    )
    split = pl.Series(["train"] * 12 + ["test"] * 12)
    levels, artifacts, summaries = [], [], []
    fit = LevelTransformer.fit

    def capture(self, frame):
        assert frame.equals(data.X.head(12))
        result = fit(self, frame)
        levels.append(self.feature_levels)
        return result

    import ml_toolkit.evaluation as evaluation

    original = evaluation.regression.observed_fitted_by_level

    def plot(*args, **kwargs):
        result = original(*args, **kwargs)
        summaries.append(result)
        return result

    monkeypatch.setattr(LevelTransformer, "fit", capture)
    monkeypatch.setattr(evaluation.regression, "observed_fitted_by_level", plot)
    monkeypatch.setattr(
        "ml_toolkit.training.reporting.tracking.log_artifact",
        lambda path, artifact_path: artifacts.append(f"{artifact_path}/{path.name}"),
    )
    reporting = ReportingConfig(
        psi=psi,
        n_bins=3,
        features=(
            FeatureLevelConfig("unused", "categorical"),
            FeatureLevelConfig("x", "numerical"),
        ),
    )
    evaluators = (ObservedFitted(),) if observed else ()
    report_predictions(
        model=MeanModel({}),
        data=data,
        prediction=np.ones(len(data)),
        split=split,
        evaluators=evaluators,
        config=reporting,
        stage="final",
    )
    assert len(levels) == int(psi) + int(observed)
    if psi and observed:
        assert levels[0] == levels[1]
    assert bool(summaries) == observed
    for summary in summaries:
        assert (
            summary.loc[summary["sample"] == "reference", "volume"].sum()
            == data.weight.head(12).sum()
        )
        assert (
            summary.loc[summary["sample"] == "current", "volume"].sum()
            == data.weight.tail(12).sum()
        )
    assert any(p.startswith("psi/") for p in artifacts) == psi
    assert any(p.startswith("observed_fitted/") for p in artifacts) == observed


def test_level_fit_never_learns_heldout_values_or_retains_previous_fit(
    tmp_path, monkeypatch
):
    data = make_data()
    evaluator = ObservedFitted(features=("x",))
    learned = []
    fit = LevelTransformer.fit

    def capture(self, frame):
        result = fit(self, frame)
        learned.append(self.feature_levels)
        return result

    monkeypatch.setattr(LevelTransformer, "fit", capture)
    inputs = dict(
        prediction=np.ones(len(data)),
        split=pl.Series(["train"] * 12 + ["val"] * 12),
        artifact_dir=tmp_path,
        figure=FigureConfig(),
        features=None,
        n_bins=3,
    )
    evaluator.write_artifacts(data=data, **inputs)
    changed = replace(
        data,
        X=data.X.with_columns(
            pl.Series("x", data.X["x"].head(12).to_list() + [100000] * 12)
        ),
    )
    evaluator.write_artifacts(data=changed, **inputs)
    assert learned[0] == learned[1]
    changed = replace(data, X=data.X.with_columns(pl.col("x") * 100))
    evaluator.write_artifacts(data=changed, **inputs)
    assert learned[2] != learned[1]
    assert evaluator.parameters() == {"features": ("x",)}


REPORTS = [
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
]


@pytest.mark.parametrize("evaluator", REPORTS, ids=lambda x: x.name)
def test_multiple_heldout_splits_rejected_before_writing(tmp_path, evaluator):
    data = make_data()
    with pytest.raises(ValueError, match="only one held-out split"):
        evaluator.write_artifacts(
            data=data,
            prediction=np.ones(len(data)),
            split=pl.Series(["train"] * 8 + ["val"] * 8 + ["test"] * 8),
            artifact_dir=tmp_path,
            figure=FigureConfig(),
            features=None,
            n_bins=3,
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("evaluator", REPORTS, ids=lambda x: x.name)
def test_train_only_reports_skip_without_fitting(tmp_path, monkeypatch, evaluator):
    monkeypatch.setattr(LevelTransformer, "fit", lambda *a: pytest.fail("fit"))
    data = make_data()
    assert (
        evaluator.write_artifacts(
            data=data,
            prediction=np.ones(len(data)),
            split=pl.Series(["train"] * len(data)),
            artifact_dir=tmp_path,
            figure=FigureConfig(),
            features=None,
            n_bins=10,
        )
        == []
    )
    assert (
        evaluator.evaluate(
            data=data,
            prediction=np.ones(len(data)),
            split=pl.Series(["train"] * len(data)),
        )
        == {}
    )


def test_disabled_levels_and_missing_features(tmp_path, monkeypatch):
    data = make_data()
    inputs = dict(
        data=data,
        prediction=np.ones(len(data)),
        split=pl.Series(["train"] * 12 + ["val"] * 12),
        artifact_dir=tmp_path,
        figure=FigureConfig(),
        n_bins=3,
    )
    assert ObservedFitted().write_artifacts(features=(), **inputs) == []
    with pytest.raises(ValueError, match="features"):
        ObservedFitted(features=("absent",)).write_artifacts(features=None, **inputs)
    with pytest.raises(ValueError, match="missing"):
        ObservedFitted().write_artifacts(
            features=(FeatureLevelConfig("missing", "numerical"),), **inputs
        )
    with pytest.raises(ValueError, match="missing"):
        NumericalFeatureDistributions(features=("missing",)).write_artifacts(
            features=None, **inputs
        )


@pytest.mark.parametrize("report", [Calibration(), Gains(), Lift()])
def test_invalid_outcomes_are_not_silently_skipped(tmp_path, report):
    data = make_data().with_target(pl.Series("y", [-1.0] * 24))
    with pytest.raises(ValueError):
        report.write_artifacts(
            data=data,
            prediction=np.ones(len(data)),
            split=pl.Series(["train"] * 12 + ["val"] * 12),
            artifact_dir=tmp_path,
            figure=FigureConfig(),
            features=None,
            n_bins=10,
        )
