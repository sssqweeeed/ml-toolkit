"""Console levels report existing results without changing training or logging."""

from contextlib import nullcontext, redirect_stderr
from contextvars import Context
from dataclasses import asdict, replace
import io
import logging

import numpy as np
import optuna
import pytest

from ml_toolkit._console import Logger, message, metrics, output_context
from ml_toolkit.config import TrainingConfig, TrackingConfig, TuningConfig
from ml_toolkit.feature_selection import RFESelector, TopKSelector, ForwardSelector
from ml_toolkit.metrics import MetricScores
from ml_toolkit.models import CatBoostRegressionModel
from ml_toolkit.tracking import runs as tracking
from ml_toolkit.training import Holdout, TimeSeriesCV, importance_ranker, train_model
from tests.support.training import MeanModel, mae, make_data

LEVELS = ("silent", "normal", "verbose", "debug")


@pytest.fixture
def recorded_tracking(monkeypatch):
    records = []
    monkeypatch.setattr(tracking, "run", lambda name: nullcontext())
    monkeypatch.setattr(
        tracking, "log_json", lambda value, path: records.append((path, value))
    )
    monkeypatch.setattr(
        tracking, "log_params", lambda value: records.append(("params", value))
    )
    monkeypatch.setattr(
        tracking, "log_metrics", lambda value: records.append(("metrics", value))
    )
    monkeypatch.setattr(
        tracking,
        "log_artifact",
        lambda path, artifact_path: records.append(
            (f"{artifact_path}/{path.name}", path.read_bytes())
        ),
    )
    return records


def run(verbosity="normal", **kwargs):
    config = kwargs.pop("config", TrainingConfig())
    return train_model(
        splitter=kwargs.pop(
            "splitter", Holdout(make_data(), val_size=0.25, test_size=0.25)
        ),
        params={"private_parameter": "do-not-print"},
        model_factory=kwargs.pop("model_factory", MeanModel),
        metric=kwargs.pop("metric", mae),
        run_name=kwargs.pop("run_name", "console-test"),
        config=replace(
            config,
            verbosity=verbosity,
            tracking=TrackingConfig(log_test_model=False, log_final_model=False),
        ),
        **kwargs,
    )


@pytest.mark.parametrize("level", LEVELS)
def test_config_defaults_replace_and_serialization(level):
    config = TrainingConfig()
    assert config.verbosity == "normal"
    changed = replace(config, verbosity=level)
    assert asdict(changed)["verbosity"] == level
    assert config.verbosity == "normal"


@pytest.mark.parametrize("value", ["info", "DEBUG", "", None, True, 2, []])
def test_invalid_verbosity(value):
    with pytest.raises(ValueError, match="silent, normal, verbose, debug"):
        TrainingConfig(verbosity=value)


@pytest.mark.parametrize("level", LEVELS)
def test_level_content_and_no_duplicate_summaries(recorded_tracking, capsys, level):
    result = run(level)
    captured = capsys.readouterr()
    assert captured.out == ""
    output = captured.err
    assert dict(recorded_tracking)["configuration/requested.json"]["verbosity"] == level
    assert "do-not-print" not in output
    assert "\x1b" not in output and "\r" not in output
    if level == "silent":
        assert output == ""
        return
    for text in (
        "Starting training run console-test",
        "Dataset: 24 rows, 2 features",
        "Starting validation split 1",
        "Completed validation split 1",
        "Starting test stage",
        "Completed test stage",
        "Evaluation summary",
        "Completed training run console-test",
    ):
        assert output.count(text) == 1
    assert "metric | validation mean | test" in output
    assert format(result.validation.mean_metrics["mae"], ".6g") in output
    assert format(result.test_metrics["mae"], ".6g") in output
    details = (
        "Validation split 1 data",
        "validation/split_1 metrics",
        "test data",
        "test metrics",
        "best iteration (zero-based): —",
        "2024-01-01",
    )
    for text in details:
        assert (text in output) == (level in ("verbose", "debug"))
    # The structured console tables replace the old raw dictionary messages.
    assert "Evaluation validation/split_1 val metrics:" not in output
    assert "Validation mean metrics:" not in output
    assert ("Refit selected iterations:" in output) == (level == "debug")


def test_levels_preserve_results_calls_and_tracking(recorded_tracking):
    results, counts, tracked = [], [], []
    for level in LEVELS:
        calls = {"fit": 0, "predict": 0, "metric": 0}

        class CountingModel(MeanModel):
            def fit(self, train, validation=None):
                calls["fit"] += 1
                return super().fit(train, validation)

            def predict_pool(self, data):
                calls["predict"] += 1
                return super().predict_pool(data)

        def metric(y, prediction, weight):
            calls["metric"] += 1
            return mae(y, prediction, weight)

        recorded_tracking.clear()
        result = run(
            level,
            model_factory=CountingModel,
            metric=metric,
            splitter=TimeSeriesCV(make_data(), n_splits=2, test_size=0.25),
            config=TrainingConfig(fit_on_test=True),
        )
        counts.append(calls.copy())
        results.append(
            (result.validation, result.test_metrics, result.params, result.features)
        )
        tracked.append(
            [
                (key, value)
                for key, value in recorded_tracking
                if key != "configuration/requested.json"
            ]
        )
        np.testing.assert_array_equal(
            result.model.predict_pool(make_data()), np.full(24, 9.5)
        )
        np.testing.assert_array_equal(
            result.final_model.predict_pool(make_data()), np.full(24, 12.5)
        )
    assert all(item == results[0] for item in results)
    assert all(item == tracked[0] for item in tracked)
    assert counts == [{"fit": 4, "predict": 7, "metric": 7}] * 4


@pytest.mark.parametrize("level", LEVELS)
def test_selection_and_tuning_levels(recorded_tracking, capsys, level):
    before = optuna.logging.get_verbosity()
    calls = {"fit": 0, "predict": 0, "metric": 0}

    class CountingModel(MeanModel):
        def fit(self, train, validation=None):
            calls["fit"] += 1
            return super().fit(train, validation)

        def predict_pool(self, data):
            calls["predict"] += 1
            return super().predict_pool(data)

    def metric(y, prediction, weight):
        calls["metric"] += 1
        return mae(y, prediction, weight)

    run(
        level,
        model_factory=CountingModel,
        metric=metric,
        feature_selector=RFESelector(
            ranker=importance_ranker,
            min_features=1,
            metric="mae",
            direction="minimize",
        ),
        config=TrainingConfig(
            tuning=TuningConfig(metric="mae", direction="minimize", n_trials=2, seed=1)
        ),
        param_space=lambda trial: {"choice": trial.suggest_int("choice", 1, 2)},
    )
    output = capsys.readouterr().err
    assert optuna.logging.get_verbosity() == before
    assert calls == {"fit": 6, "predict": 10, "metric": 10}
    assert output.count("Completed feature selection:") == int(level != "silent")
    assert output.count("Completed Optuna tuning:") == int(level != "silent")
    assert output.count("Completed tuning trial") == (
        2 if level in ("verbose", "debug") else 0
    )
    assert output.count("Selected features:") == int(level == "debug")
    assert ("Selection candidate:" in output) == (level == "debug")
    assert "do-not-print" not in output


@pytest.mark.parametrize("level", LEVELS)
def test_model_diagnostics_and_native_output_settings(recorded_tracking, capsys, level):
    params = {
        "iterations": 2,
        "depth": 2,
        "thread_count": 1,
        "random_seed": 1,
        "verbose": False,
        "allow_writing_files": False,
    }
    result = train_model(
        splitter=Holdout(make_data(), val_size=0.25),
        params=params,
        model_factory=CatBoostRegressionModel,
        metric=mae,
        run_name="native-settings",
        config=TrainingConfig(
            verbosity=level,
            tracking=TrackingConfig(log_test_model=False, log_final_model=False),
        ),
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert ("Fitted CatBoost regressor" in captured.err) == (level == "debug")
    assert ("Predicting with CatBoost regressor" in captured.err) == (level == "debug")
    assert result.model.native_model.get_params()["verbose"] is False
    assert params["verbose"] is False


@pytest.mark.parametrize(
    "selector",
    [
        TopKSelector(k=1, ranker=importance_ranker),
        ForwardSelector(metric="mae", direction="minimize", max_features=1, max_fits=2),
    ],
)
def test_other_selectors_have_summary_and_debug_steps(
    recorded_tracking, capsys, selector
):
    run("debug", feature_selector=selector)
    output = capsys.readouterr().err
    assert output.count("Completed feature selection:") == 1
    assert "Selection candidate:" in output
    assert output.count("Selected features:") == 1


def test_no_test_and_additional_final_fit(recorded_tracking, capsys):
    run("verbose", splitter=Holdout(make_data(), val_size=0.25))
    output = capsys.readouterr().err
    assert "Starting final_fit stage" in output
    assert "final_fit metrics" in output
    assert "test" not in output.replace("console-test", "")
    run("verbose", config=TrainingConfig(fit_on_test=True))
    output = capsys.readouterr().err
    assert output.index("Completed test stage") < output.index(
        "Starting final_fit stage"
    )
    assert "final_fit data (including test)" in output
    assert output.count("test scores belong to the main model") == 1


def test_empty_and_nonfinite_existing_metrics(recorded_tracking, capsys):
    run(metric=lambda y, p, w: MetricScores({}))
    assert (
        "Evaluation summary (test scores belong to the main model): no metrics"
        in capsys.readouterr().err
    )
    run(
        metric=lambda y, p, w: MetricScores(
            {"undefined": float("nan"), "unbounded": float("inf")}
        )
    )
    output = capsys.readouterr().err
    assert "nan" in output and "inf" in output
    with output_context("normal"):
        metrics(
            "Different keys",
            {"validation": {"a": 1.0}, "test": {"b": 2.0}},
            verbosity="normal",
        )
    assert "—" in capsys.readouterr().err


def test_console_independent_of_logging_and_restores_after_failure(
    recorded_tracking, caplog
):
    logger = logging.getLogger("ml_toolkit")
    caplog.set_level(logging.DEBUG, logger="ml_toolkit")
    state = (logger.level, tuple(logger.handlers), logger.propagate)
    stream = io.StringIO()
    with redirect_stderr(stream):
        run("silent")
    assert stream.getvalue() == ""
    assert any(
        record.getMessage() == "Starting training run console-test"
        for record in caplog.records
    )

    class BrokenModel(MeanModel):
        def fit(self, train, validation=None):
            raise RuntimeError("fit failed")

    with redirect_stderr(stream), pytest.raises(RuntimeError, match="fit failed"):
        run("debug", model_factory=BrokenModel)
    assert "Completed training run" not in stream.getvalue()
    stream = io.StringIO()
    with redirect_stderr(stream):
        Logger("ml_toolkit.outside").info("outside the run")
        run("silent")
    assert stream.getvalue() == ""
    assert (logger.level, tuple(logger.handlers), logger.propagate) == state
    assert any(record.getMessage() == "outside the run" for record in caplog.records)


def test_nested_runs_restore_outer_verbosity(recorded_tracking, capsys):
    class NestedModel(MeanModel):
        def fit(self, train, validation=None):
            if validation is not None:
                run("silent", run_name="nested")
                message("outer debug restored", verbosity="debug")
            return super().fit(train, validation)

    run("debug", model_factory=NestedModel)
    output = capsys.readouterr().err
    assert "nested" not in output
    assert output.count("outer debug restored") == 1
    message("outside")
    assert capsys.readouterr().err == ""


def test_independent_contexts_and_warnings(capsys):
    quiet, loud = Context(), Context()
    quiet_scope = quiet.run(output_context, "silent")
    loud_scope = loud.run(output_context, "debug")
    quiet.run(quiet_scope.__enter__)
    loud.run(loud_scope.__enter__)
    try:
        quiet.run(message, "quiet")
        loud.run(message, "loud", verbosity="debug")
        message("outside")
        assert capsys.readouterr().err == "loud\n"
        with pytest.warns(UserWarning, match="still visible"):
            import warnings

            quiet.run(warnings.warn, "still visible")
    finally:
        quiet.run(quiet_scope.__exit__, None, None, None)
        loud.run(loud_scope.__exit__, None, None, None)
