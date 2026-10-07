"""Configuration contracts and observable workflow behavior."""

from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
from datetime import date
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
import inspect
import re

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import polars as pl
import pytest

import ml_toolkit.config as settings
from ml_toolkit.config import (
    ClassificationDiagnosticsConfig,
    ClassificationPlotsConfig,
    CommonPlotsConfig,
    CrossValidationConfig,
    RFEFeatureSelectionConfig,
    FigureConfig,
    FinalFitConfig,
    NumericalFeatureDistributionConfig,
    PostFitEvaluationConfig,
    PUDiagnosticsConfig,
    PUPlotsConfig,
    PSIOptions,
    RegressionDiagnosticsConfig,
    RegressionPlotsConfig,
    ShapConfig,
    TrackingConfig,
    TrainingConfig,
    TuningConfig,
)
from ml_toolkit import Pool
from ml_toolkit.explainability import calculate_shap_values
from ml_toolkit.models import CatBoostRegressionModel, InflationAdjustedModel
from ml_toolkit.inflation import ConstantRateInflation
from ml_toolkit.plots.classification import calibration, precision_recall
from ml_toolkit.plots.monitoring import monthly_feature_distribution
from ml_toolkit.plots.utils import save_figure
from ml_toolkit.psi import PSIFeature
from ml_toolkit.training import importance_ranker, train_model
from ml_toolkit.training.cross_validation import _time_series_splits
from test_training import MeanModel, list_all_artifacts, mae, make_data, setup_mlflow


@pytest.mark.parametrize(
    "factory",
    [
        lambda: CrossValidationConfig(n_splits=True),
        lambda: TuningConfig(metric="mae", direction="invalid"),
        lambda: FigureConfig(figsize=(float("nan"), 5)),
        lambda: ShapConfig(seed=False),
        lambda: CommonPlotsConfig(psi_scores=1),
        lambda: PUDiagnosticsConfig(plots=RegressionPlotsConfig()),
        lambda: ClassificationDiagnosticsConfig(plots=PUPlotsConfig()),
        lambda: RegressionDiagnosticsConfig(task="pu_classification"),
        lambda: TrainingConfig(cv={"n_splits": 3}),
        lambda: NumericalFeatureDistributionConfig(features=["x"]),
    ],
)
def test_invalid_types_and_task_combinations_fail_on_construction(factory):
    with pytest.raises(TypeError):
        factory()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: CrossValidationConfig(n_splits=1),
        lambda: CrossValidationConfig(n_splits=3, skip_splits=3),
        lambda: CrossValidationConfig(n_splits=3, gap=-1),
        lambda: RFEFeatureSelectionConfig(min_features=0, metric_name="mae", direction="minimize"),
        lambda: RFEFeatureSelectionConfig(min_features=1, metric_name="mae", direction="minimize", validation_fraction=1),
        lambda: TuningConfig(metric="", direction="minimize"),
        lambda: TuningConfig(metric="mae", direction="minimize", timeout_seconds=0),
        lambda: PSIOptions(epsilon=0),
        lambda: PSIOptions(features=()),
        lambda: FigureConfig(figsize=(0, 5)),
        lambda: ShapConfig(dependence_features=("x", "x")),
        lambda: PUDiagnosticsConfig(folds=()),
        lambda: PUDiagnosticsConfig(folds=(0,)),
        lambda: NumericalFeatureDistributionConfig(quantile_bands=((0.9, 0.1),)),
        lambda: PostFitEvaluationConfig(stages=("cv", "cv")),
        lambda: TrainingConfig(
            cv=CrossValidationConfig(n_splits=3, skip_splits=1),
            diagnostics=PUDiagnosticsConfig(folds=(1,)),
        ),
    ],
)
def test_invalid_ranges_fail_on_construction(factory):
    with pytest.raises(ValueError):
        factory()


def test_configs_are_immutable_keyword_only_and_replaceable():
    original = TrainingConfig(cv=CrossValidationConfig(n_splits=3))
    with pytest.raises(FrozenInstanceError):
        original.cv.gap = 7
    changed = replace(original, cv=replace(original.cv, gap=7))
    assert original.cv.gap == 0 and changed.cv.gap == 7
    with pytest.raises(TypeError):
        CrossValidationConfig(3)
    assert not hasattr(original, "__dict__")
    assert "gains" not in {f.name for f in fields(PUPlotsConfig)}
    assert PUDiagnosticsConfig().task == "pu_classification"
    assert ClassificationDiagnosticsConfig().task == "classification"


def test_config_docstrings_describe_every_public_field():
    for name in settings.__all__:
        cls = getattr(settings, name)
        if not inspect.isclass(cls) or not is_dataclass(cls):
            continue
        doc = inspect.getdoc(cls)
        assert "Parameters\n----------" in doc, name
        for item in fields(cls):
            assert re.search(rf"^{item.name} :", doc, re.MULTILINE), (name, item.name)


def test_markdown_has_one_executable_fully_expanded_recipe_per_task():
    document = Path(__file__).resolve().parents[1] / "docs/configuration.md"
    blocks = re.findall(r"```python\n(.*?)\n```", document.read_text(), re.DOTALL)
    namespace = {}
    exec(blocks[0], namespace)
    recipes = blocks[1:6]
    assert len(recipes) == 5
    import ast

    for block in recipes:
        tree = ast.parse(block)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                cls = namespace[node.func.id]
                assert {kw.arg for kw in node.keywords} == {
                    f.name for f in fields(cls)
                }, cls
        exec(block, namespace)
    assert namespace["pu_classification_config"].feature_selection.min_features == 10
    assert namespace["severity_config"].diagnostics.plots.gains
    assert not namespace["regression_config"].diagnostics.plots.gains


def test_cv_honors_gap_and_row_window_sizes():
    config = CrossValidationConfig(n_splits=3, gap=2, test_size=4, max_train_size=5)
    splits = list(_time_series_splits(24, config))
    assert len(splits) == 3
    for index, (train, validation) in splits:
        assert len(train) == 5 and len(validation) == 4
        assert validation[0] - train[-1] == 3


@pytest.mark.parametrize("classifier", [False, True])
def test_selection_fits_native_model_with_normal_parameters(classifier):
    from ml_toolkit.models import CatBoostClassificationModel
    from ml_toolkit.training.feature_selection import select_features
    data = make_data()
    if classifier:
        data = data.with_target(pl.Series("target", [0, 1] * 12))
    fitted = []
    factory = CatBoostClassificationModel if classifier else CatBoostRegressionModel
    def record(params):
        model = factory(params)
        fitted.append(model)
        return model
    params = {"iterations": 3, "verbose": False, "allow_writing_files": False}
    result = select_features(metric=mae, ranker=importance_ranker, data=data, params=params, model_factory=record, config=RFEFeatureSelectionConfig(min_features=1, metric_name="mae", direction="minimize"))
    assert [len(step.features) for step in result.steps] == [2, 1]
    assert result.features == result.steps[result.selected_step].features
    assert fitted[0].native_model.tree_count_ > 0
    assert fitted[0].params == params
    assert not hasattr(fitted[0], "select_features")



def test_selection_fraction_and_cv_gap_reach_workflow(monkeypatch, tmp_path):
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

    selection = RFEFeatureSelectionConfig(min_features=1, metric_name="mae", direction="minimize", validation_fraction=0.5, gap=1)
    config = TrainingConfig(
        cv=CrossValidationConfig(n_splits=2, gap=2), feature_selection=selection
    )
    train_model(
        data=make_data(),
        params={},
        model_factory=Model,
        metric=mae,
        feature_ranker=importance_ranker,
        run_name="split-options",
        config=config,
    )
    assert len(fits[0][0]) == 2 and len(fits[0][1]) == 3
    assert fits[0][0][-1] == 1 and fits[0][1][0] == 3
    assert fits[1] == fits[0]
    assert fits[2][0][-1] == 5 and fits[2][1][0] == 8


@pytest.mark.parametrize("policy, expected", [("cv_median", 4), ("provided", 20)])
def test_final_iteration_policy_and_model_packaging(tmp_path, policy, expected):
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
        log_fold_models=True,
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
            cv=CrossValidationConfig(n_splits=2),
            tracking=tracking,
            final_fit=FinalFitConfig(iteration_policy=policy),
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


def test_standalone_model_logging_uses_example_size(monkeypatch):
    data = make_data()
    model = CatBoostRegressionModel(
        {"iterations": 2, "depth": 2, "verbose": False, "allow_writing_files": False}
    ).fit(data)
    examples = []
    monkeypatch.setattr(
        model,
        "log_artifacts",
        lambda frame, predictions, **kwargs: examples.append((frame, kwargs)),
    )
    monkeypatch.setattr(mlflow, "log_param", lambda *a, **k: None)
    monkeypatch.setattr(mlflow, "log_dict", lambda *a, **k: None)
    model.log(data, config=TrackingConfig(input_example_rows=2, model_name="two-rows"))
    assert len(examples[0][0]) == 2 and examples[0][1]["name"] == "two-rows"


def test_inflation_wrapper_uses_shared_importance_selection():
    from ml_toolkit.training.feature_selection import select_features
    data = make_data()
    fitted = []
    def factory(params):
        model = InflationAdjustedModel(CatBoostRegressionModel(params), ConstantRateInflation(0.03), date(2024, 1, 1))
        fitted.append(model)
        return model
    result = select_features(metric=mae, ranker=importance_ranker, data=data, params={"iterations": 3, "verbose": False, "allow_writing_files": False}, model_factory=factory, config=RFEFeatureSelectionConfig(min_features=1, metric_name="mae", direction="minimize"))
    assert [len(step.features) for step in result.steps] == [2, 1]
    assert result.features == result.steps[result.selected_step].features
    assert fitted[0].native_model.tree_count_ > 0
    assert not hasattr(fitted[0], "select_features")



@pytest.mark.parametrize(
    "folds, diagnosed",
    [("last", ["fold_3"]), ("all", ["fold_1", "fold_2", "fold_3"]), ((2,), ["fold_2"])],
)
def test_diagnostics_only_run_for_selected_folds(
    tmp_path, monkeypatch, folds, diagnosed
):
    setup_mlflow(tmp_path)
    calls = []
    monkeypatch.setattr(
        "ml_toolkit.training.evaluation._log_diagnostics",
        lambda *args: calls.append(args[-1]),
    )
    train_model(
        data=make_data(),
        params={},
        model_factory=MeanModel,
        metric=mae,
        run_name="fold-policy",
        config=TrainingConfig(
            cv=CrossValidationConfig(n_splits=3),
            diagnostics=RegressionDiagnosticsConfig(folds=folds, plots=None),
        ),
    )
    assert calls == diagnosed


def test_disabled_diagnostics_do_not_repeat_tuned_cv(tmp_path):
    setup_mlflow(tmp_path)
    fits = []

    class Model(MeanModel):
        def fit(self, train, validation=None):
            fits.append(len(train))
            return super().fit(train, validation)

    result = train_model(
        data=make_data(),
        params={},
        model_factory=Model,
        metric=mae,
        run_name="no-repeated-cv",
        param_space=lambda trial: {"fixed": 42},
        config=TrainingConfig(
            cv=CrossValidationConfig(n_splits=2),
            tuning=TuningConfig(metric="mae", direction="minimize", n_trials=1, seed=7),
            diagnostics=RegressionDiagnosticsConfig(psi=None, plots=None),
        ),
    )
    assert fits == [8, 16, 24]
    assert result.params["fixed"] == 42


@pytest.mark.parametrize(
    "stages, splits, images, expected",
    [
        (("cv", "final"), ("val",), False, [("fold_1", "val"), ("fold_2", "val")]),
        (("tuning",), ("val",), True, [("fold_1", "val"), ("fold_2", "val")]),
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

    evaluator = SimpleNamespace(evaluate=evaluate, write_artifacts=writer)
    train_model(
        data=data,
        params={},
        model_factory=MeanModel,
        metric=mae,
        run_name="post-policy",
        param_space=lambda trial: {},
        post_fit_evaluator=evaluator,
        config=TrainingConfig(
            cv=CrossValidationConfig(n_splits=2),
            tuning=TuningConfig(metric="mae", direction="minimize", n_trials=1),
            post_fit=PostFitEvaluationConfig(
                stages=stages, splits=splits, write_artifacts=images
            ),
        ),
    )
    assert calls == expected
    assert len(written) == (len(expected) if images else 0)


def test_disabled_post_fit_does_not_require_ids(tmp_path):
    setup_mlflow(tmp_path)
    evaluator = SimpleNamespace(
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
            cv=CrossValidationConfig(n_splits=2),
            post_fit=PostFitEvaluationConfig(stages=()),
        ),
    )


def test_psi_definition_and_svg_output_reach_diagnostics(tmp_path):
    setup_mlflow(tmp_path)
    plots = RegressionPlotsConfig(
        common=CommonPlotsConfig(
            figure=FigureConfig(format="svg", figsize=(6, 4), dpi=90),
            psi_distribution_features=("x",),
            numerical_distributions=None,
        ),
        monthly_observed_fitted=False,
    )
    train_model(
        data=make_data(),
        params={},
        model_factory=MeanModel,
        metric=mae,
        run_name="psi-options",
        config=TrainingConfig(
            cv=CrossValidationConfig(n_splits=2),
            diagnostics=RegressionDiagnosticsConfig(
                psi=PSIOptions(
                    n_bins=2,
                    epsilon=0.01,
                    features=(PSIFeature("x", "numerical", lower_bound=0),),
                ),
                plots=plots,
            ),
        ),
    )
    runs = mlflow.search_runs(output_format="list")
    for run in runs:
        artifacts = list_all_artifacts(mlflow.MlflowClient(), run.info.run_id)
        if run.data.tags["mlflow.runName"] == "fold_2":
            assert "figures/fold_2/psi_scores.svg" in artifacts
            assert any(path.endswith("psi_000_x.svg") for path in artifacts)
            table = Path(
                mlflow.artifacts.download_artifacts(
                    run_id=run.info.run_id, artifact_path="psi/fold_2/scores.json"
                )
            )
            import json

            assert [
                row["feature"] for row in json.loads(table.read_text())["rows"]
            ] == ["x"]
        else:
            assert not any(path.startswith("psi/") for path in artifacts)


def test_custom_quantile_bands_and_hidden_curves(tmp_path):
    data = make_data()
    frame = data.X.with_columns(data.time)
    config = NumericalFeatureDistributionConfig(
        quantile_bands=((0.1, 0.9),), show_mean=False, show_median=False
    )
    summary = monthly_feature_distribution(
        frame, frame, "x", "date", tmp_path / "quantiles.png", config=config
    )
    assert (
        "p10" in summary.columns
        and "p90" in summary.columns
        and "p05" not in summary.columns
    )
    assert (tmp_path / "quantiles.png").is_file()


def test_figure_export_honors_format_dimensions_and_dpi(tmp_path, monkeypatch):
    fig, _ = plt.subplots()
    calls = []
    save = fig.savefig

    def capture(path, **kwargs):
        calls.append((tuple(fig.get_size_inches()), kwargs))
        return save(path, **kwargs)

    monkeypatch.setattr(fig, "savefig", capture)
    result = save_figure(
        fig,
        tmp_path / "plot.png",
        config=FigureConfig(figsize=(4, 3), dpi=72, format="svg"),
    )
    assert result == tmp_path / "plot.svg" and result.is_file()
    assert calls[0][0] == (4, 3) and calls[0][1]["dpi"] == 72
    assert not plt.fignum_exists(fig.number)


def test_classification_resolution_preserves_full_curve(tmp_path, monkeypatch):
    cfg = ClassificationPlotsConfig(
        max_curve_points=3, calibration_bins=2, histogram_bins=4
    )
    y = np.array([0, 1] * 10)
    p = np.linspace(0.01, 0.99, 20)
    captured = []

    def capture(fig, path, **kwargs):
        captured.append(fig)

    monkeypatch.setattr("ml_toolkit.plots.classification.save_figure", capture)
    try:
        curve = precision_recall(y, p, tmp_path / "pr.png", config=cfg)
        assert len(curve) == 20 and len(captured[0].axes[0].lines[0].get_xdata()) == 4
        groups = calibration(y, p, tmp_path / "cal.png", config=cfg)
        assert len(groups) == 2 and len(captured[1].axes[1].patches) == 4
    finally:
        for fig in captured:
            plt.close(fig)


def test_shap_sampling_honors_limit_and_seed():
    pytest.importorskip("shap")
    data = make_data()
    model = CatBoostRegressionModel(
        {"iterations": 2, "depth": 2, "verbose": False, "allow_writing_files": False}
    ).fit(data)
    cfg = ShapConfig(max_samples=5, seed=23)
    first = calculate_shap_values(model.native_model, data.X, config=cfg)
    second = calculate_shap_values(model.native_model, data.X, config=cfg)
    other = calculate_shap_values(
        model.native_model, data.X, config=replace(cfg, seed=24)
    )
    assert first.values.shape == (5, 2)
    np.testing.assert_array_equal(first.data, second.data)
    assert not np.array_equal(first.data, other.data)


def test_tuning_budget_seed_and_trial_packaging(tmp_path, monkeypatch):
    setup_mlflow(tmp_path)
    module = import_module("ml_toolkit.training.tuning")
    studies, budgets, logged = [], [], []
    real_create = module.optuna.create_study

    def create(**kwargs):
        study = real_create(**kwargs)
        studies.append(kwargs)
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

    train_model(
        data=make_data(),
        params={},
        model_factory=Model,
        metric=mae,
        run_name="trial-packaging",
        param_space=lambda trial: {"choice": trial.suggest_int("choice", 1, 3)},
        config=TrainingConfig(
            cv=CrossValidationConfig(n_splits=2),
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
    assert budgets == [{"n_trials": 2, "timeout": 60}]
    assert studies[0]["sampler"]._rng.rng.randint(10000) == np.random.RandomState(
        19
    ).randint(10000)
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
            cv=CrossValidationConfig(n_splits=2),
            diagnostics=RegressionDiagnosticsConfig(),
        ),
    )
    assert np.all(result.model.predict_pool(data) < 0)
    runs = mlflow.search_runs(output_format="list")
    fold = next(run for run in runs if run.data.tags["mlflow.runName"] == "fold_2")
    paths = list_all_artifacts(mlflow.MlflowClient(), fold.info.run_id)
    assert any(path.endswith("monthly_observed_fitted.png") for path in paths)
    assert not any(path.endswith(("gains.png", "lift.png")) for path in paths)
