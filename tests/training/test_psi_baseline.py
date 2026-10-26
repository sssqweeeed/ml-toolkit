"""Each logged stage packages its own full training reference, regardless of reports."""

from itertools import product

import mlflow
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from examples.custom_selector import FixedFeatureSelector

from ml_toolkit.config import (
    ReportingConfig,
    TrackingConfig,
    TrainingConfig,
    TuningConfig,
)
from ml_toolkit.feature_levels import LevelTransformer
from ml_toolkit.psi import PSIBaseline
from ml_toolkit.training import Holdout, train_model
from ml_toolkit.training.psi import prepare_psi_baseline, log_model_with_baseline
from tests.support.training import MeanModel, mae, make_data, setup_mlflow


@pytest.mark.parametrize("psi,logs,reports", list(product((False, True), repeat=3)))
def test_stage_matrix(tmp_path, monkeypatch, psi, logs, reports):
    setup_mlflow(tmp_path)
    data = make_data()
    splitter = Holdout(data, val_size=4, test_size=4)
    saved, trained = {}, {}
    fits = []
    original_fit = LevelTransformer.fit

    def fit_levels(self, reference):
        fits.append(reference.clone())
        return original_fit(self, reference)

    monkeypatch.setattr(LevelTransformer, "fit", fit_levels)

    class CapturingModel(MeanModel):
        def fit(self, train, validation=None):
            self.train = train
            trained[mlflow.active_run().data.tags["mlflow.runName"]] = train
            return super().fit(train, validation)

        def log(self, sample, *, config=None, **kwargs):
            name = mlflow.active_run().data.tags["mlflow.runName"]
            if not psi:
                assert kwargs == {}  # Preserve calls to preexisting adapters.
                saved[name] = None
            else:
                baseline = PSIBaseline.load(kwargs["extra_artifacts"]["psi_baseline"])
                assert baseline.metadata["n_rows"] == len(self.train)
                assert baseline.metadata["n_rows"] > config.input_example_rows
                saved[name] = baseline

    train_model(
        splitter=splitter,
        params={},
        model_factory=CapturingModel,
        metric=mae,
        run_name="baseline-stages",
        param_space=lambda trial: {"choice": trial.suggest_int("choice", 1, 2)},
        config=TrainingConfig(
            fit_on_test=True,
            tuning=TuningConfig(metric="mae", direction="minimize", n_trials=1),
            reporting=ReportingConfig(
                psi=psi,
                on_tuning=reports,
                on_validation=reports,
                on_test=reports,
                on_final_fit=reports,
            ),
            tracking=TrackingConfig(
                log_trial_models=logs,
                log_validation_models=logs,
                log_test_model=logs,
                log_final_model=logs,
                input_example_rows=1,
            ),
        ),
    )
    assert len(saved) == (4 if logs else 0)
    if psi and logs:
        assert len(fits) == 4  # Reports reuse the transformer instead of fitting again.
        assert saved["test"].metadata["n_rows"] == len(splitter.get_test().train)
        assert saved["final_fit"].metadata["n_rows"] == len(splitter.get_final())
        for name, baseline in saved.items():
            assert_frame_equal(
                baseline.compare(trained[name].X).psi.select("psi"),
                pl.DataFrame({"psi": [0.0] * len(data.X.columns)}),
            )
    if not psi or (not logs and not reports):
        assert fits == []


@pytest.mark.parametrize("enabled", ["trial", "validation", "test", "final"])
def test_independent_model_flags(tmp_path, enabled):
    setup_mlflow(tmp_path)
    saved = []

    class CapturingModel(MeanModel):
        def log(self, sample, *, config=None, extra_artifacts=None):
            baseline = PSIBaseline.load(extra_artifacts["psi_baseline"])
            saved.append(baseline.metadata["stage"])

    train_model(
        splitter=Holdout(make_data(), val_size=4, test_size=4),
        params={},
        model_factory=CapturingModel,
        metric=mae,
        run_name="independent-flags",
        param_space=lambda trial: {},
        config=TrainingConfig(
            fit_on_test=True,
            tuning=TuningConfig(metric="mae", direction="minimize", n_trials=1),
            reporting=ReportingConfig(
                psi=True,
                on_tuning=False,
                on_validation=False,
                on_test=False,
                on_final_fit=False,
            ),
            tracking=TrackingConfig(
                log_trial_models=enabled == "trial",
                log_validation_models=enabled == "validation",
                log_test_model=enabled == "test",
                log_final_model=enabled == "final",
            ),
        ),
    )
    assert saved == [{"trial": "tuning", "final": "final_fit"}.get(enabled, enabled)]


def test_selected_features_disabled_levels_and_save_failure(monkeypatch):
    data = make_data().select_features(("x",))
    baseline = prepare_psi_baseline(
        data, ReportingConfig(psi=True), log_model=True, stage="final_fit"
    )
    assert [f.name for f in baseline.transformer.features] == ["x"]
    assert (
        prepare_psi_baseline(
            data, ReportingConfig(psi=True, features=()), log_model=True, stage="test"
        )
        is None
    )

    class Model:
        def log(self, *args, **kwargs):
            pytest.fail("Must not log or register after baseline save fails")

    def fail_save(*args, **kwargs):
        raise OSError("storage failure")

    monkeypatch.setattr(PSIBaseline, "save", fail_save)
    with pytest.raises(OSError, match="storage failure"):
        log_model_with_baseline(Model(), data, TrackingConfig(), baseline)


def test_final_fit_without_test(tmp_path):
    setup_mlflow(tmp_path)
    saved = []

    class CapturingModel(MeanModel):
        def log(self, sample, *, config=None, extra_artifacts=None):
            saved.append(PSIBaseline.load(extra_artifacts["psi_baseline"]))

    data = make_data()
    train_model(
        splitter=Holdout(data, val_size=4),
        params={},
        model_factory=CapturingModel,
        feature_selector=FixedFeatureSelector(("x",)),
        metric=mae,
        run_name="train-only",
        config=TrainingConfig(
            reporting=ReportingConfig(psi=True, on_validation=False, on_final_fit=False)
        ),
    )
    assert len(saved) == 1
    assert saved[0].metadata["stage"] == "final_fit"
    assert saved[0].metadata["n_rows"] == len(data)
    assert tuple(f.name for f in saved[0].transformer.features) == ("x",)
