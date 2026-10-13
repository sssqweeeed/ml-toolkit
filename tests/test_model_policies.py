from ml_toolkit.config import HoldoutConfig
from ml_toolkit.training import split_holdout

"""Baseline policies, shared importance selection and portable serving state."""

import json
import os
import subprocess
import sys
import sysconfig
import venv
from datetime import date, timedelta
from pathlib import Path

import catboost
import mlflow
import numpy as np
import pandas as pd
import polars as pl
import pytest
from test_training import mae, setup_mlflow

from ml_toolkit import Pool
from ml_toolkit.config import HoldoutConfig, RFEFeatureSelectionConfig
from ml_toolkit.inflation import ConstantRateInflation, ManualInflation
from ml_toolkit.models import (
    CatBoostClassificationModel,
    CatBoostLogLinkRegressionModel,
    CatBoostPUClassificationModel,
    CatBoostRegressionModel,
    InflationAdjustedModel,
)
from ml_toolkit.training import ShapRanker, importance_ranker
from ml_toolkit.training.evaluation import calculate_metrics
from ml_toolkit.training.feature_selection import select_features
from ml_toolkit.types import FloatArray

PARAMS = dict(
    iterations=5,
    depth=2,
    learning_rate=0.1,
    thread_count=1,
    verbose=False,
    allow_writing_files=False,
)


@pytest.fixture(scope="session")
def inference_python(tmp_path_factory: pytest.TempPathFactory) -> Path:
    environment = tmp_path_factory.mktemp("inference-environment")
    builder = venv.EnvBuilder(with_pip=False)
    builder.create(environment)
    python = Path(builder.ensure_directories(environment).env_exe)
    library = Path(
        subprocess.check_output(
            [
                str(python),
                "-I",
                "-c",
                "import sysconfig; print(sysconfig.get_path('purelib'))",
            ],
            text=True,
        ).strip()
    )
    # Reuse installed runtime libraries without installing the toolkit or training
    # dependencies. Exclude .pth files so editable installs cannot expose source.
    for entry in Path(sysconfig.get_path("purelib")).iterdir():
        if entry.suffix == ".pth" or entry.name.startswith(
            ("optuna", "numba", "llvmlite", "ml_toolkit", "__editable__")
        ):
            continue
        (library / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
    return python


def check_bundled_inference(
    python: Path,
    tmp_path: Path,
    directory: Path,
    frame: pd.DataFrame,
    expected: FloatArray,
    kind: str,
    nested: str = "",
) -> None:
    requirements = (directory / "requirements.txt").read_text().lower()
    assert not any(
        line.startswith(("optuna", "numba", "ml-toolkit", "ml_toolkit"))
        for line in requirements.splitlines()
    )
    frame.to_pickle(tmp_path / "frame.pkl")
    np.save(tmp_path / "prediction.npy", expected)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            str(python),
            "-I",
            str(Path(__file__).with_name("check_model_inference.py")),
            str(tmp_path),
            mlflow.get_tracking_uri(),
            str(directory),
            kind,
            nested,
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture
def data():
    rng = np.random.default_rng(42)
    n = 48
    return Pool(
        pl.DataFrame(
            {
                "noise": rng.normal(size=n),
                "signal": np.arange(n, dtype=float),
                "category": [10, None, 20] * 16,
            }
        ),
        pl.Series("target", 1.0 + np.arange(n) / 8),
        pl.Series("date", [date(2024, 1, 1) + timedelta(days=i * 5) for i in range(n)]),
        pl.Series("weight", np.linspace(0.5, 2, n)),
        cat_features=("category",),
    )


class SquaredObjective:
    def calc_ders_range(self, approxes, targets, weights):
        return [
            (
                (target - approx) * (1.0 if weights is None else weights[i]),
                -(1.0 if weights is None else weights[i]),
            )
            for i, (approx, target) in enumerate(zip(approxes, targets))
        ]


@pytest.mark.parametrize(
    "loss", [None, "RMSE", "MAE", "Quantile:alpha=0.7", SquaredObjective()]
)
def test_general_regressor_preserves_native_initialization_and_outputs(data, loss):
    data = data.with_target(data.y - 8)
    params = {**PARAMS, "boost_from_average": False}
    if loss is not None:
        params["loss"] = loss
    if isinstance(loss, SquaredObjective):
        params["eval_metric"] = "RMSE"
        params["leaf_estimation_method"] = "Gradient"
    model = CatBoostRegressionModel(params).fit(data)
    native_params = {key: value for key, value in params.items() if key != "loss"}
    if loss is not None:
        native_params["loss_function"] = loss
    native = catboost.CatBoostRegressor(**native_params).fit(data.to_catboost())
    assert model.baseline is None
    assert model.task == "regression"
    assert model.native_model.get_params()["boost_from_average"] is False
    np.testing.assert_allclose(
        model.predict_pool(data), native.predict(data.to_model_input())
    )
    assert (model.predict_pool(data) < 0).any()
    assert params == model.params


def test_native_boost_from_average_is_preserved(data):
    model = CatBoostRegressionModel({**PARAMS, "boost_from_average": True}).fit(
        data.with_target(data.y - 10)
    )
    assert model.baseline is None
    assert model.native_model.get_all_params()["boost_from_average"] is True
    assert model.native_model.get_scale_and_bias()[1] != 0


@pytest.mark.parametrize("loss", ["Poisson", "Tweedie:variance_power=1.5"])
def test_log_link_baseline_shared_by_splits_and_folded_once(data, loss):
    from ml_toolkit.models import catboost_regression as module

    train, validation = data.take(range(32)), data.take(range(32, 48))
    baseline = float(
        np.log(np.average(train.y.to_numpy(), weights=train.weight.to_numpy()))
    )
    model = CatBoostLogLinkRegressionModel(
        {**PARAMS, "loss": loss, "boost_from_average": True}
    ).fit(train, validation)
    native = catboost.CatBoostRegressor(
        **PARAMS, loss_function=loss, boost_from_average=False
    )
    native.fit(
        train.to_catboost(baseline=baseline),
        eval_set=validation.to_catboost(baseline=baseline),
    )
    scale, bias = native.get_scale_and_bias()
    assert model.baseline == pytest.approx(baseline)
    assert model.native_model.get_scale_and_bias() == pytest.approx(
        (scale, bias + baseline)
    )
    native.set_scale_and_bias(scale, bias + baseline)
    np.testing.assert_allclose(
        model.predict_pool(data), native.predict(data.to_model_input())
    )
    assert model.native_model.get_all_params()["boost_from_average"] is False
    assert (
        module.CatBoostLogLinkRegressionModel.fit is module.CatBoostRegressionModel.fit
    )


@pytest.mark.parametrize("loss", ["RMSE", "MAE", SquaredObjective()])
def test_loss_restriction_belongs_only_to_log_link(data, loss):
    with pytest.raises(ValueError, match="Poisson or Tweedie"):
        CatBoostLogLinkRegressionModel({**PARAMS, "loss": loss}).fit(data)


@pytest.mark.parametrize("target", [0.0, -1.0, float("nan")])
def test_log_link_rejects_invalid_weighted_mean(data, target):
    with pytest.raises(ValueError, match="finite and positive"):
        CatBoostLogLinkRegressionModel(PARAMS).fit(
            data.with_target(np.full(len(data), target))
        )


@pytest.mark.parametrize(
    "factory",
    [
        CatBoostRegressionModel,
        CatBoostLogLinkRegressionModel,
        CatBoostClassificationModel,
        CatBoostPUClassificationModel,
    ],
)
def test_loss_aliases_reject_conflicts(factory, data):
    with pytest.raises(ValueError, match="Conflicting"):
        factory({**PARAMS, "loss": "Poisson", "loss_function": "RMSE"}).fit(data)


def test_prediction_type_and_multidimensional_output_are_preserved(data):
    model = CatBoostRegressionModel(PARAMS, prediction_type="RawFormulaVal").fit(data)
    np.testing.assert_allclose(
        model.predict_pool(data),
        model.native_model.predict(
            data.to_model_input(), prediction_type="RawFormulaVal"
        ),
    )

    class Native:
        feature_names_ = ["signal"]

        def get_cat_feature_indices(self):
            return []

        def predict(self, frame, **options):
            return np.zeros((len(frame), 2))

    model._model = Native()
    prediction = model.predict_pool(data)
    assert prediction.shape == (len(data), 2)
    with pytest.raises(ValueError, match="one finite value per row"):
        calculate_metrics(mae, data, prediction)


def test_supervised_prevalence_and_population_prior_are_distinct(data):
    data = data.with_target(pl.Series("target", [1, 0, 0] * 16))
    supervised = CatBoostClassificationModel(PARAMS).fit(data)
    pu = CatBoostPUClassificationModel({**PARAMS, "pu_prior": 0.12}).fit(data)
    prevalence = np.average(data.y.to_numpy(), weights=data.weight.to_numpy())
    assert supervised.baseline == pytest.approx(np.log(prevalence / (1 - prevalence)))
    assert pu.baseline == pytest.approx(np.log(0.12 / 0.88))
    assert pu.pu_prior == 0.12 and pu.pu_loss == "uPU"
    assert pu.task == "pu_classification" and supervised.task == "classification"
    assert CatBoostPUClassificationModel.fit is CatBoostClassificationModel.fit
    with pytest.raises(ValueError, match="CatBoostPUClassificationModel"):
        CatBoostClassificationModel({**PARAMS, "pu_prior": 0.12}).fit(data)
    with pytest.raises(ValueError, match="Logloss or CrossEntropy"):
        CatBoostClassificationModel({**PARAMS, "loss": "uPU"}).fit(data)


@pytest.mark.parametrize("loss", ["Logloss", "CrossEntropy"])
def test_supervised_threshold_policy_and_refit_reset(data, loss):
    binary = data.with_target(pl.Series("target", [1, 0, 0] * 16))
    train, validation = binary.take(range(32)), binary.take(range(32, 48))
    model = CatBoostClassificationModel({**PARAMS, "loss": loss}).fit(train, validation)
    expected_threshold, expected_f1 = model._select_threshold(
        validation.y.to_numpy(), model.predict_pool(validation)
    )
    assert model.threshold == expected_threshold and model.validation_f1 == expected_f1
    model.fit(binary)
    assert model.threshold == 0.5 and model.validation_f1 is None
    explicit = CatBoostClassificationModel({**PARAMS, "decision_threshold": 0.8}).fit(
        train, validation
    )
    assert explicit.threshold == 0.8 and explicit.validation_f1 is None


@pytest.mark.parametrize("loss", ["uPU", "nnPU"])
def test_pu_threshold_policy_and_refit(data, loss):
    binary = data.with_target(pl.Series("target", [1, 0, 0] * 16))
    model = CatBoostPUClassificationModel({**PARAMS, "loss": loss, "pu_prior": 0.2})
    model.fit(binary.take(range(32)), binary.take(range(32, 48)))
    assert model.validation_f1 is None
    assert np.isfinite(model.predict_pool(binary)).all()
    with pytest.raises(RuntimeError, match="decision_threshold"):
        model.predict_labels_pool(binary)
    model.params["decision_threshold"] = 0.4
    model.fit(binary)
    assert model.threshold == 0.4 and model.validation_f1 is None
    assert model.best_iteration is None


def test_nnpu_correction_branch_early_stopping_and_best_tree(monkeypatch, data):
    import ml_toolkit.models.catboost_classification as module

    binary = data.with_target(pl.Series("target", [0, 1] * 24))
    calls, risks = [], iter([3.0, 2.0, 4.0, 5.0])
    original_risks = module._pu_risks
    # Control training versus validation separately; retain the actual native fits.
    train, val = binary.take(range(32)), binary.take(range(32, 48))

    def risk(logits, labels, weights, prior):
        original_risks(logits, labels, weights, prior)
        if len(labels) == 32:
            return 1.0, -1.0
        if len(calls) == 0:
            return 1.0, 1.0
        return 0.0, next(risks)

    fit = catboost.CatBoostClassifier.fit

    def record(self, *args, **kwargs):
        objective = self.get_params()["loss_function"]
        calls.append(objective.reverse_negative_risk)
        return fit(self, *args, **kwargs)

    monkeypatch.setattr(module, "_pu_risks", risk)
    monkeypatch.setattr(catboost.CatBoostClassifier, "fit", record)
    model = CatBoostPUClassificationModel(
        {
            **PARAMS,
            "iterations": 10,
            "loss": "nnPU",
            "pu_prior": 0.7,
            "early_stopping_rounds": 2,
            "eval_metric": "Logloss",
        }
    ).fit(train, val)
    assert calls == [True] * 4
    assert model.best_iteration == 1 and model.native_model.tree_count_ == 2


@pytest.mark.parametrize(
    "kind", ["regression", "log_link", "classification", "uPU", "nnPU", "inflation"]
)
@pytest.mark.parametrize(
    "ranker", [importance_ranker, ShapRanker(max_samples=5, seed=42)]
)
def test_rfe_selection_preserves_adapter_policies_and_original_order(
    data, kind, ranker
):
    if kind in ("classification", "uPU", "nnPU"):
        data = data.with_target(pl.Series("target", [1, 0, 0] * 16))
    fitted = []

    def factory(params):
        if kind == "regression":
            model = CatBoostRegressionModel(params)
        elif kind == "log_link":
            model = CatBoostLogLinkRegressionModel(params)
        elif kind == "classification":
            model = CatBoostClassificationModel(params)
        elif kind in ("uPU", "nnPU"):
            model = CatBoostPUClassificationModel(
                {**params, "loss": kind, "pu_prior": 0.2}
            )
        else:
            model = InflationAdjustedModel(
                CatBoostLogLinkRegressionModel(params),
                ConstantRateInflation(0.03),
                date(2024, 1, 1),
            )
        fitted.append(model)
        return model

    from ml_toolkit.metrics import PUMetric

    metric = PUMetric(0.2, loss=kind) if kind in ("uPU", "nnPU") else mae
    metric_name = "pu_risk" if kind in ("uPU", "nnPU") else "mae"
    partition = split_holdout(data.take(range(32)), HoldoutConfig(val_size=0.25))
    result = select_features(
        metric=metric,
        ranker=ranker,
        train=partition.train,
        validation=partition.val,
        params=PARAMS,
        model_factory=factory,
        config=RFEFeatureSelectionConfig(
            min_features=2, metric=metric_name, direction="minimize"
        ),
    )
    assert len(fitted) == len(result.steps) == 2
    assert [len(step.features) for step in result.steps] == [3, 2]
    assert result.features == result.steps[result.selected_step].features
    assert list(result.features) == [
        name for name in data.X.columns if name in result.features
    ]
    assert all(np.isfinite(step.metrics[metric_name]) for step in result.steps)


@pytest.mark.parametrize(
    "kind",
    [
        "regression",
        "custom",
        "log_link",
        "classification",
        "uPU",
        "nnPU",
        "inflation_native",
        "inflation_log_link",
    ],
)
def test_models_from_code_roundtrip_outside_repository(
    data, tmp_path, kind, inference_python
):
    setup_mlflow(tmp_path)
    if kind in ("classification", "uPU", "nnPU"):
        data = data.with_target(pl.Series("target", [1, 0, 0] * 16))
    if kind == "regression":
        model = CatBoostRegressionModel(PARAMS, prediction_type="RawFormulaVal")
    elif kind == "custom":
        model = CatBoostRegressionModel(
            {
                **PARAMS,
                "loss": SquaredObjective(),
                "eval_metric": "RMSE",
                "leaf_estimation_method": "Gradient",
            },
        )
    elif kind == "log_link":
        model = CatBoostLogLinkRegressionModel(PARAMS)
    elif kind == "classification":
        model = CatBoostClassificationModel({**PARAMS, "decision_threshold": 0.4})
    elif kind in ("uPU", "nnPU"):
        model = CatBoostPUClassificationModel(
            {**PARAMS, "loss": kind, "pu_prior": 0.2, "decision_threshold": 0.3},
        )
    else:
        regressor_factory = (
            CatBoostRegressionModel
            if kind == "inflation_native"
            else CatBoostLogLinkRegressionModel
        )
        regressor = regressor_factory(PARAMS)
        inflation = (
            ConstantRateInflation(0.03)
            if kind == "inflation_native"
            else ManualInflation({date(2024, 1, 1): 1.0, date(2025, 1, 1): 1.1})
        )
        model = InflationAdjustedModel(regressor, inflation, date(2024, 1, 1))
    model.fit(data.take(range(32)), data.take(range(32, 48)))
    frame = data.to_model_input()
    # Nonstandard index and shuffled feature order must not disturb dates or categories.
    frame.index = pd.Index(np.arange(len(frame)) + 100)
    frame = frame.loc[:, list(reversed(frame.columns))]
    if isinstance(model, InflationAdjustedModel):
        frame[model.prediction_date_column] = pd.Series(
            data.time.to_list(), index=frame.index, dtype="datetime64[ns]"
        )
    expected = model.predict_pool(data)
    np.testing.assert_allclose(model.predict_frame(frame), expected)
    with mlflow.start_run():
        info = model.log(data)
    if isinstance(info, tuple):
        info = info[1]
    directory = Path(mlflow.artifacts.download_artifacts(info.model_uri))
    files = list((directory / "code").rglob("*"))
    assert not any(
        path.name in ("__pycache__", ".codex", ".codegraph", "AGENTS.md")
        for path in files
    )
    state = model.export_state()
    assert state["version"] == 1 and state["model_type"] == type(model).__name__
    nested = (
        type(model.model).__name__ if isinstance(model, InflationAdjustedModel) else ""
    )
    check_bundled_inference(
        inference_python,
        tmp_path,
        directory,
        frame,
        expected,
        type(model).__name__,
        nested,
    )


def test_importing_model_modules_does_not_register_models():
    code = "import mlflow; calls=[]; mlflow.models.set_model=calls.append; import ml_toolkit.models; assert calls == []"
    result = subprocess.run(
        [sys.executable, "-c", code], text=True, capture_output=True, timeout=60
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("manual", [False, True])
def test_inflation_preserves_datetime_precision_and_validates_dates(data, manual):
    from dataclasses import replace
    from datetime import datetime

    timestamps = pl.Series(
        "date",
        [
            datetime(2024, 1, 1, 13, 30) + timedelta(days=i * 5)
            for i in range(len(data))
        ],
    )
    data = replace(data, time=timestamps)
    strategy = (
        ManualInflation({date(2024, 1, 1): 1.0, date(2025, 1, 1): 1.1})
        if manual
        else ConstantRateInflation(0.03)
    )
    model = InflationAdjustedModel(
        CatBoostRegressionModel(PARAMS), strategy, date(2024, 1, 1)
    ).fit(data)
    expected = strategy.adjust(
        model.model.predict_pool(data), model._reference_dates(len(data)), data.time
    )
    np.testing.assert_allclose(model.predict_pool(data), expected, rtol=1e-14)
    frame = data.X.with_columns(data.time.alias(model.prediction_date_column))
    np.testing.assert_allclose(model.predict_frame(frame), expected, rtol=1e-14)
    with pytest.raises(ValueError, match="Missing model input column"):
        model.predict_frame(data.to_model_input())
    invalid = frame.to_pandas()
    invalid.loc[0, model.prediction_date_column] = pd.NaT
    with pytest.raises(ValueError, match="must not contain null"):
        model.predict_frame(invalid)


def test_inflation_selection_fits_strategy_on_supplied_training_only(data):
    seen = []

    class RecordingInflation(ConstantRateInflation):
        def fit(self, dates, y, sample_weight=None):
            seen.append((dates.clone(), y.copy(), sample_weight.copy()))
            return super().fit(dates, y, sample_weight)

    partition = split_holdout(data.take(range(32)), HoldoutConfig(val_size=0.25))
    select_features(
        metric=mae,
        ranker=importance_ranker,
        train=partition.train,
        validation=partition.val,
        params={},
        model_factory=lambda params: InflationAdjustedModel(
            CatBoostRegressionModel(PARAMS),
            RecordingInflation(0.03),
            date(2024, 1, 1),
        ),
        config=RFEFeatureSelectionConfig(
            min_features=2,
            metric="mae",
            direction="minimize",
        ),
    )
    assert len(seen) == 2
    assert seen[0][0].to_list() == data.time.head(24).to_list()
    np.testing.assert_array_equal(seen[0][1], data.y.head(24).to_numpy())
    np.testing.assert_array_equal(seen[0][2], data.weight.head(24).to_numpy())


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), "invalid"])
def test_both_classifiers_validate_explicit_threshold(data, threshold):
    binary = data.with_target(pl.Series("target", [1, 0, 0] * 16))
    for factory, extra in [
        (CatBoostClassificationModel, {}),
        (CatBoostPUClassificationModel, {"pu_prior": 0.2}),
    ]:
        with pytest.raises(ValueError, match="decision_threshold"):
            factory({**PARAMS, **extra, "decision_threshold": threshold}).fit(binary)


@pytest.mark.parametrize(
    "extra, message",
    [
        ({}, "requires pu_prior"),
        ({"pu_prior": 0.0}, "strictly between"),
        ({"pu_prior": float("nan")}, "strictly between"),
        ({"pu_prior": 0.2, "loss": "nnPU", "task_type": "GPU"}, "CPU"),
        ({"pu_prior": 0.2, "task_type": "TPU"}, "task_type"),
        ({"pu_prior": 0.2, "task_type": "GPU", "devices": "0:1"}, "one device"),
        ({"pu_prior": 0.2, "class_weights": [1, 2]}, "incompatible"),
    ],
)
def test_pu_constraints_are_local_to_pu(data, extra, message):
    binary = data.with_target(pl.Series("target", [1, 0, 0] * 16))
    with pytest.raises(ValueError, match=message):
        CatBoostPUClassificationModel({**PARAMS, **extra}).fit(binary)


def test_pu_objective_derivatives_and_risks_are_stable():
    from ml_toolkit.models.catboost_classification import (
        _pu_risks,
        _PULogisticObjective,
    )

    logits = np.array([-1000.0, 1000.0, -1000.0, 1000.0])
    labels = np.array([1, 1, 0, 0])
    weights = np.ones(4)
    objective = _PULogisticObjective(0.5, positive_scale=2.0, unlabeled_scale=3.0)
    np.testing.assert_allclose(
        objective.calc_ders_range(logits, labels, weights),
        [[2.0, 0.0], [2.0, 0.0], [0.0, 0.0], [-3.0, 0.0]],
    )
    objective.reverse_negative_risk = True
    np.testing.assert_allclose(
        objective.calc_ders_range(logits, labels, weights),
        [[0.0, 0.0], [-2.0, 0.0], [0.0, 0.0], [3.0, 0.0]],
    )
    assert np.isfinite(_pu_risks(logits, labels, weights, 0.5)).all()


def test_loading_rejects_wrong_state_version_or_policy(data, tmp_path):
    from types import SimpleNamespace

    model = CatBoostLogLinkRegressionModel(PARAMS).fit(data)
    path = tmp_path / "model.cbm"
    model.native_model.save_model(str(path))
    state = model.export_state()
    config = tmp_path / "config.json"
    for update in [
        {"version": 2},
        {"model_type": "CatBoostRegressionModel"},
        {"baseline_mode": "raw_prediction"},
    ]:
        config.write_text(json.dumps({**state, **update}))
        with pytest.raises(ValueError, match="state|baseline"):
            CatBoostLogLinkRegressionModel().load_context(
                SimpleNamespace(artifacts={"model": str(path), "config": str(config)})
            )


def test_optional_lightgbm_models_from_code(data, tmp_path, inference_python):
    lightgbm = pytest.importorskip("lightgbm")
    from ml_toolkit.models.lightgbm import LightGBMWrapper

    setup_mlflow(tmp_path)
    frame = data.X.select("noise", "signal").to_pandas()
    estimator = lightgbm.LGBMRegressor(n_estimators=3, verbosity=-1, n_jobs=1).fit(
        frame, data.y.to_numpy()
    )
    model = LightGBMWrapper(estimator, numerical_features=list(frame.columns))
    with mlflow.start_run():
        info = model.log(data)
    loaded = mlflow.pyfunc.load_model(info.model_uri)
    assert type(loaded.unwrap_python_model()).__name__ == "LightGBMWrapper"
    np.testing.assert_allclose(loaded.predict(frame), model.predict(None, frame))
    directory = Path(mlflow.artifacts.download_artifacts(info.model_uri))
    check_bundled_inference(
        inference_python,
        tmp_path,
        directory,
        frame,
        model.predict(None, frame),
        "LightGBMWrapper",
    )
