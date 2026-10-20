"""Observable training and reporting configuration policies."""

from dataclasses import replace
from importlib import import_module
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
import pytest
from tests.support.evaluators import CallbackEvaluator
from tests.support.training import (
    MeanModel,
    list_all_artifacts,
    mae,
    make_data,
    setup_mlflow,
)

import ml_toolkit.config as settings
from examples.project_pool import ProjectPool as Pool
from ml_toolkit.feature_selection import RFESelector
from ml_toolkit.config import (
    CommonPlotsConfig,
    FigureConfig,
    PostFitEvaluationConfig,
    RegressionDiagnosticsConfig,
    RegressionPlotsConfig,
    ReportingConfig,
    TimeSeriesCVConfig,
    TrackingConfig,
    TrainingConfig,
    TuningConfig,
)
from ml_toolkit.feature_levels import FeatureLevelSpec
from ml_toolkit.training import importance_ranker, train_model
from ml_toolkit.training.splits import make_validation_splits


def test_cv_honors_gap_and_row_window_sizes():
    config = TimeSeriesCVConfig(n_splits=3, gap=2, val_size=4, max_train_size=5)
    splits = make_validation_splits(make_data(), config)
    assert len(splits) == 3
    for part in splits:
        assert len(part.train) == 5 and len(part.val) == 4
        assert part.val.X["x"][0] - part.train.X["x"][-1] == 3


def test_selection_uses_external_train_val_and_gap(monkeypatch, tmp_path):
    setup_mlflow(tmp_path)
    fits = []

    class Model(MeanModel):
        def fit(self, train, validation=None):
            fits.append(
                (
                    train.X["x"].to_list(),
                    None if validation is None else validation.X["x"].to_list(),
                )
            )
            return super().fit(train, validation)

    selection = RFESelector(
        ranker=importance_ranker,
        min_features=1,
        metric="mae",
        direction="minimize",
    )
    config = TrainingConfig(validation=TimeSeriesCVConfig(n_splits=2, gap=2))
    train_model(
        data=make_data(),
        params={},
        model_factory=Model,
        metric=mae,
        run_name="split-options",
        config=config,
        feature_selector=replace(selection, ranker=importance_ranker),
    )
    assert len(fits[0][0]) == 6 and len(fits[0][1]) == 8
    assert fits[0][0][-1] == 5 and fits[0][1][0] == 8
    assert fits[1] == fits[0]
    assert fits[2][0][-1] == 5 and fits[2][1][0] == 8


def test_final_iteration_and_model_packaging(tmp_path):
    expected = 4
    setup_mlflow(tmp_path)
    logged, fitted = [], []

    class Model(MeanModel):
        @property
        def best_iteration(self):
            return 3

        def fit(self, train, validation=None):
            fitted.append(self.params.copy())
            return super().fit(train, validation)

        def log(self, sample, *, config=None):
            logged.append((len(sample), config))

    tracking = TrackingConfig(
        log_validation_models=True,
        input_example_rows=2,
        model_name="custom",
        registered_model_name="registry",
    )
    params = {"iterations": 20, "early_stopping_rounds": 2, "use_best_model": True}
    result = train_model(
        data=make_data(),
        params=params,
        model_factory=Model,
        metric=mae,
        run_name="final-policy",
        config=TrainingConfig(
            validation=TimeSeriesCVConfig(n_splits=2), tracking=tracking
        ),
    )
    assert result.params["iterations"] == expected
    assert params["iterations"] == 20 and params["use_best_model"] is True
    assert (
        "early_stopping_rounds" not in fitted[-1]
        and fitted[-1]["use_best_model"] is False
    )
    assert [cfg.registered_model_name for _, cfg in logged] == [None, None, "registry"]
    assert all(
        cfg.input_example_rows == 2 and cfg.model_name == "custom" for _, cfg in logged
    )
    parent = next(
        run
        for run in mlflow.search_runs(output_format="list")
        if run.data.tags["mlflow.runName"] == "final-policy"
    )
    artifacts = list_all_artifacts(mlflow.MlflowClient(), parent.info.run_id)
    assert (
        "configuration/requested.json" in artifacts
        and "configuration/resolved.json" in artifacts
    )


@pytest.mark.parametrize(
    "folds, diagnosed",
    [
        ("last", ["validation/split_3"]),
        ("all", ["validation/split_1", "validation/split_2", "validation/split_3"]),
        ((2,), ["validation/split_2"]),
    ],
)
def test_diagnostics_only_run_for_selected_folds(
    tmp_path, monkeypatch, folds, diagnosed
):
    setup_mlflow(tmp_path)
    calls = []
    monkeypatch.setattr(
        "ml_toolkit.training.evaluation.report_diagnostics",
        lambda *args: calls.append(args[-2]),
    )
    train_model(
        data=make_data(),
        params={},
        model_factory=MeanModel,
        metric=mae,
        run_name="fold-policy",
        config=TrainingConfig(
            validation=TimeSeriesCVConfig(n_splits=3),
            reporting=ReportingConfig(
                diagnostics=RegressionDiagnosticsConfig(plots=None),
                validation_splits=folds,
            ),
        ),
    )
    assert calls == diagnosed


@pytest.mark.parametrize(
    "stages, splits, images, expected",
    [
        (
            ("validation", "final"),
            ("val",),
            False,
            [("validation/split_1", "val"), ("validation/split_2", "val")],
        ),
        (
            ("tuning",),
            ("val",),
            True,
            [("trial_1", "val"), ("trial_1", "val")],
        ),
        (("final",), ("train",), True, [("final_fit", "train")]),
    ],
)
def test_post_fit_stage_split_and_image_policies(
    tmp_path, stages, splits, images, expected
):
    setup_mlflow(tmp_path)
    source = make_data()
    data = Pool(
        source.X,
        source.y,
        source.time,
        source.weight,
        object_id=pl.Series("id", list(range(24))),
    )
    calls, written = [], []

    def evaluate(**kwargs):
        calls.append(
            (mlflow.active_run().data.tags["mlflow.runName"], kwargs["split"].value)
        )
        return {}

    def writer(**kwargs):
        written.append(kwargs["split"].value)
        return []

    evaluator = CallbackEvaluator(evaluate=evaluate, write_artifacts=writer)
    train_model(
        data=data,
        params={},
        model_factory=MeanModel,
        metric=mae,
        run_name="post-policy",
        param_space=lambda trial: {},
        post_fit_evaluator=evaluator,
        config=TrainingConfig(
            validation=TimeSeriesCVConfig(n_splits=2),
            tuning=TuningConfig(metric="mae", direction="minimize", n_trials=1),
            reporting=ReportingConfig(
                post_fit=PostFitEvaluationConfig(
                    stages=stages, splits=splits, write_artifacts=images
                ),
            ),
        ),
    )
    assert calls == expected
    assert len(written) == (len(expected) if images else 0)


def test_disabled_post_fit_does_not_require_ids(tmp_path):
    setup_mlflow(tmp_path)
    evaluator = CallbackEvaluator(
        evaluate=lambda **kwargs: pytest.fail("disabled evaluator")
    )
    train_model(
        data=make_data(),
        params={},
        model_factory=MeanModel,
        metric=mae,
        run_name="disabled-post",
        post_fit_evaluator=evaluator,
        config=TrainingConfig(
            validation=TimeSeriesCVConfig(n_splits=2),
            reporting=ReportingConfig(post_fit=PostFitEvaluationConfig(stages=())),
        ),
    )


def test_psi_definition_and_svg_output_reach_diagnostics(tmp_path):
    setup_mlflow(tmp_path)
    plots = RegressionPlotsConfig(
        common=CommonPlotsConfig(
            figure=FigureConfig(format="svg", figsize=(6, 4), dpi=90),
            n_bins=2,
            epsilon=0.01,
            features=(FeatureLevelSpec("x", "numerical", lower_bound=0),),
            numerical_distributions=None,
        ),
        time_observed_fitted=False,
    )
    train_model(
        data=make_data(),
        params={},
        model_factory=MeanModel,
        metric=mae,
        run_name="psi-options",
        config=TrainingConfig(
            validation=TimeSeriesCVConfig(n_splits=2),
            reporting=ReportingConfig(
                diagnostics=RegressionDiagnosticsConfig(
                    plots=plots,
                ),
            ),
        ),
    )
    runs = mlflow.search_runs(output_format="list")
    for run in runs:
        artifacts = list_all_artifacts(mlflow.MlflowClient(), run.info.run_id)
        if run.data.tags["mlflow.runName"] == "validation/split_2":
            assert set(artifacts) == {
                "figures/psi/scores.svg",
                "figures/psi/time_by_feature/000_x.svg",
                "figures/observed_fitted/by_feature/000_x.svg",
                "tables/psi/scores.json",
                "tables/psi/level_distribution.json",
                "tables/psi/time_distribution.json",
            }
            table = Path(
                mlflow.artifacts.download_artifacts(
                    run_id=run.info.run_id, artifact_path="tables/psi/scores.json"
                )
            )
            import json

            assert [
                row["feature"] for row in json.loads(table.read_text())["rows"]
            ] == ["x"]
        else:
            assert not any(path.startswith("tables/psi/") for path in artifacts)


def test_tuning_budget_seed_and_trial_packaging(tmp_path, monkeypatch):
    setup_mlflow(tmp_path)
    module = import_module("ml_toolkit.training.tuning")
    studies, budgets, logged = [], [], []
    suggestions = []
    real_create = module.optuna.create_study

    def create(**kwargs):
        study = real_create(**kwargs)
        studies.append(study)
        real_optimize = study.optimize

        def optimize(objective, **settings):
            budgets.append(settings)
            return real_optimize(objective, **settings)

        monkeypatch.setattr(study, "optimize", optimize)
        return study

    monkeypatch.setattr(module.optuna, "create_study", create)

    class Model(MeanModel):
        def log(self, sample, *, config=None):
            logged.append(config)

    def param_space(trial):
        value = trial.suggest_float("choice", 0.0, 1.0)
        suggestions.append(value)
        return {"choice": value}

    train_model(
        data=make_data(),
        params={},
        model_factory=Model,
        metric=mae,
        run_name="trial-packaging",
        param_space=param_space,
        config=TrainingConfig(
            validation=TimeSeriesCVConfig(n_splits=2),
            tuning=TuningConfig(
                metric="mae",
                direction="minimize",
                n_trials=2,
                timeout_seconds=60,
                seed=19,
            ),
            tracking=TrackingConfig(
                log_trial_models=True,
                log_final_model=False,
                registered_model_name="final-only",
            ),
        ),
    )
    assert budgets == [{"n_trials": 2, "timeout": 60}] * 2
    assert len(studies) == 2
    # Independent split searches with the same seed reproduce suggestions.
    assert suggestions[:2] == suggestions[2:]
    assert len(set(suggestions[:2])) == 2
    assert len(logged) == 4 and all(
        config.registered_model_name is None for config in logged
    )


def test_ordinary_regression_defaults_accept_signed_targets(tmp_path):
    setup_mlflow(tmp_path)
    source = make_data()
    data = source.with_target(source.y - 100)
    result = train_model(
        data=data,
        params={},
        model_factory=MeanModel,
        metric=mae,
        run_name="signed-regression",
        config=TrainingConfig(
            validation=TimeSeriesCVConfig(n_splits=2),
            reporting=ReportingConfig(diagnostics=RegressionDiagnosticsConfig()),
        ),
    )
    assert np.all(result.model.predict_pool(data) < 0)
    runs = mlflow.search_runs(output_format="list")
    fold = next(
        run for run in runs if run.data.tags["mlflow.runName"] == "validation/split_2"
    )
    paths = list_all_artifacts(mlflow.MlflowClient(), fold.info.run_id)
    assert "figures/observed_fitted/time.png" in paths
    assert not any(path.endswith(("gains.png", "lift.png")) for path in paths)
