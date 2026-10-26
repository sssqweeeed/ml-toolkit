"""Observable training and reporting configuration policies."""

from ml_toolkit.training import TimeSeriesCV

from dataclasses import replace
from importlib import import_module
import mlflow
from tests.support.training import (
    MeanModel,
    list_all_artifacts,
    mae,
    make_data,
    setup_mlflow,
)
import ml_toolkit.config as settings
from ml_toolkit.feature_selection import RFESelector
from ml_toolkit.config import TrackingConfig, TrainingConfig, TuningConfig
from ml_toolkit.training import importance_ranker, train_model


def test_cv_honors_gap_and_row_window_sizes():
    splitter = TimeSeriesCV(
        make_data(), n_splits=3, gap=2, val_size=4, max_train_size=5
    )
    splits = list(splitter.get_validation())
    assert len(splits) == 3
    for part in splits:
        assert len(part.train) == 5 and len(part.heldout) == 4
        assert part.heldout.X["x"][0] - part.train.X["x"][-1] == 3


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
    config = TrainingConfig()
    train_model(
        splitter=TimeSeriesCV(make_data(), n_splits=2, gap=2),
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

        def log(self, sample, *, config=None, extra_artifacts=None):
            logged.append((len(sample), config))

    tracking = TrackingConfig(
        log_validation_models=True,
        input_example_rows=2,
        model_name="custom",
        registered_model_name="registry",
    )
    params = {"iterations": 20, "early_stopping_rounds": 2, "use_best_model": True}
    result = train_model(
        splitter=TimeSeriesCV(make_data(), n_splits=2),
        params=params,
        model_factory=Model,
        metric=mae,
        run_name="final-policy",
        config=TrainingConfig(tracking=tracking),
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
        def log(self, sample, *, config=None, extra_artifacts=None):
            logged.append(config)

    def param_space(trial):
        value = trial.suggest_float("choice", 0.0, 1.0)
        suggestions.append(value)
        return {"choice": value}

    train_model(
        splitter=TimeSeriesCV(make_data(), n_splits=2),
        params={},
        model_factory=Model,
        metric=mae,
        run_name="trial-packaging",
        param_space=param_space,
        config=TrainingConfig(
            tuning=TuningConfig(
                metric="mae",
                direction="minimize",
                n_trials=2,
                timeout_seconds=60,
                seed=19,
            ),
            tracking=TrackingConfig(
                log_trial_models=True,
                log_test_model=False,
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
