"""Shared training behavior across holdout, temporal validation and tuning."""

from ml_toolkit.feature_selection import RFESelector
import json
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_series_equal
from tests.support.evaluators import CallbackEvaluator
from tests.support.training import (
    MeanModel,
    list_all_artifacts,
    mae,
    make_data,
    setup_mlflow,
)

from examples.project_pool import ProjectPool as Pool
from ml_toolkit.config import (
    HoldoutConfig,
    CommonPlotsConfig,
    RegressionPlotsConfig,
    RegressionDiagnosticsConfig,
    ReportingConfig,
    TimeSeriesCVConfig,
    TrackingConfig,
    TrainingConfig,
    TuningConfig,
)
from ml_toolkit.inflation import ConstantRateInflation
from ml_toolkit.metrics import metrics_classification
from ml_toolkit.models import (
    CatBoostClassificationModel,
    CatBoostLogLinkRegressionModel,
    InflationAdjustedModel,
)
from ml_toolkit.models._catboost import refit_parameters
from ml_toolkit.training import DatasetSplit, train_model


from tests.support.training import make_identified_data as identified_data


@pytest.mark.parametrize(
    "validation", [HoldoutConfig(val_size=0.25), TimeSeriesCVConfig(n_splits=2)]
)
@pytest.mark.parametrize("tuning", [False, True])
def test_each_split_selects_once_before_tuning_and_excludes_test(
    tmp_path, validation, tuning
):
    setup_mlflow(tmp_path)
    data = identified_data()
    fits, ranked = [], []

    class Model(MeanModel):
        def fit(self, train, validation=None):
            assert max(train.object_id) < 20
            if validation is not None:
                assert max(validation.object_id) < 20
                assert train.X.columns == validation.X.columns
            for rows in [train] + ([] if validation is None else [validation]):
                assert rows.time.to_list() == data.time.gather(rows.object_id).to_list()
                assert (
                    rows.weight.to_list()
                    == data.weight.gather(rows.object_id).to_list()
                )
                assert (
                    rows.business_type.to_list()
                    == data.business_type.gather(rows.object_id).to_list()
                )
                assert set(rows.numerical_features) == set(rows.X.columns)
            fits.append(
                (
                    train.object_id.to_list(),
                    None if validation is None else validation.object_id.to_list(),
                    tuple(train.X.columns),
                ),
            )
            self._best = None if validation is None else len(train)
            self._threshold = self.params.get("decision_threshold", len(train) / 100)
            return super().fit(train, validation)

        @property
        def best_iteration(self):
            return self._best

        def get_refit_params(self):
            params = super().get_refit_params()
            params["decision_threshold"] = self._threshold
            return params

    def ranker(model, rows):
        ranked.append(rows.object_id.to_list())
        later = min(rows.object_id) >= 10
        return {"x": 0.0 if later else 1.0, "unused": 1.0 if later else 0.0}

    params = {"iterations": 100, "early_stopping_rounds": 2, "use_best_model": True}
    config = TrainingConfig(
        validation=validation,
        test_size=4,
        tuning=(
            TuningConfig(metric="mae", direction="minimize", n_trials=2)
            if tuning
            else None
        ),
        tracking=TrackingConfig(log_final_model=False),
    )
    result = train_model(
        data=data,
        params=params,
        model_factory=Model,
        metric=mae,
        config=config,
        run_name="shared",
        param_space=lambda trial: {"fixed": 42},
        feature_selector=RFESelector(
            ranker=ranker, min_features=1, metric="mae", direction="minimize"
        ),
    )
    count = 1 if isinstance(validation, HoldoutConfig) else 2
    assert len(ranked) == count
    assert len(fits) == count * (5 if tuning else 3) + 1
    assert fits[-1] == (list(range(20)), None, result.features)
    assert result.features == ("unused",)
    assert result.params == result.validation.last_split.refit_params
    assert (
        result.params["iterations"] == result.validation.last_split.best_iteration + 1
    )
    assert result.params["decision_threshold"] == (15 if count == 1 else 14) / 100
    assert "early_stopping_rounds" not in result.params
    assert result.params["use_best_model"] is False
    assert result.test_metrics is not None
    assert params == {
        "iterations": 100,
        "early_stopping_rounds": 2,
        "use_best_model": True,
    }
    assert all(not hasattr(split, "model") for split in result.validation.splits)
    assert result.validation.mean_metrics["mae"] == pytest.approx(
        np.mean([split.val_metrics["mae"] for split in result.validation.splits]),
    )
    if count == 2:
        assert result.validation.splits[0].features == ("x",)
        assert result.validation.last_split.features == ("unused",)
    if tuning:
        assert result.tuning.best_params["fixed"] == result.params["fixed"] == 42


@pytest.mark.parametrize(
    "dtype",
    [
        pl.Date,
        pl.Datetime("ms"),
        pl.Datetime("us"),
        pl.Datetime("ns"),
        pl.Datetime("us", "Europe/Moscow"),
    ],
)
@pytest.mark.parametrize(
    "cutoff, test_ids",
    [
        ("2025-01-02", [3, 4, 5]),
        (date(2025, 1, 2), [3, 4, 5]),
        ("2025-01-03", [5]),
    ],
)
def test_calendar_test_preserves_rows_and_logs_date(tmp_path, dtype, cutoff, test_ids):
    setup_mlflow(tmp_path)
    timestamps = pl.Series(
        "time",
        [
            datetime(2025, 1, 1, 0, 15),
            datetime(2025, 1, 1, 12),
            datetime(2025, 1, 1, 23, 59),
            datetime(2025, 1, 2, 0, 15),
            datetime(2025, 1, 2, 23, 59),
            datetime(2025, 1, 4, 0, 15),
        ],
    )
    if dtype == pl.Date:
        timestamps = timestamps.cast(dtype)
    else:
        timestamps = timestamps.dt.replace_time_zone(dtype.time_zone).cast(dtype)
    data = replace(
        identified_data().take(range(6)),
        time=timestamps,
        weight=pl.Series("weight", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        cat_features=("unused",),
        numerical_features=("x",),
    )
    fit_rows, test_predictions = [], []
    train_ids = list(range(test_ids[0]))

    class Model(MeanModel):
        def fit(self, train, validation=None):
            assert set(train.object_id.to_list()) <= set(train_ids)
            if validation is not None:
                assert set(validation.object_id.to_list()) <= set(train_ids)
            fit_rows.append(train.object_id.to_list())
            return super().fit(train, validation)

        def predict_pool(self, rows):
            ids = rows.object_id.to_list()
            assert_frame_equal(rows.X, data.X[ids])
            assert_series_equal(rows.object_id, data.object_id.gather(ids))
            assert_series_equal(rows.time, data.time.gather(ids))
            assert_series_equal(rows.y, data.y.gather(ids))
            assert_series_equal(rows.weight, data.weight.gather(ids))
            assert_series_equal(rows.business_type, data.business_type.gather(ids))
            assert rows.cat_features == data.cat_features
            assert rows.numerical_features == data.numerical_features
            if ids[0] in test_ids:
                assert fit_rows[-1] == train_ids
                test_predictions.append(ids)
            return super().predict_pool(rows)

    result = train_model(
        data=data,
        params={},
        model_factory=Model,
        metric=mae,
        run_name="calendar-test",
        config=TrainingConfig(
            validation=HoldoutConfig(val_size=1),
            test_size=cutoff,
            tracking=TrackingConfig(log_final_model=False),
        ),
    )
    assert fit_rows[-1] == train_ids
    assert test_predictions == [test_ids]
    assert result.test_metrics is not None
    expected_mean = np.average(
        data.y.gather(train_ids).to_numpy(),
        weights=data.weight.gather(train_ids).to_numpy(),
    )
    expected_mae = mae(
        data.y.gather(test_ids).to_numpy(),
        np.full(len(test_ids), expected_mean),
        data.weight.gather(test_ids).to_numpy(),
    ).to_dict()["mae"]
    assert result.test_metrics["mae"] == pytest.approx(expected_mae)
    parent = next(
        run
        for run in mlflow.search_runs(output_format="list")
        if run.data.tags["mlflow.runName"] == "calendar-test"
    )
    artifact = mlflow.MlflowClient().download_artifacts(
        parent.info.run_id, "configuration/requested.json"
    )
    assert json.loads(Path(artifact).read_text())["test_size"] == str(cutoff)


@pytest.mark.parametrize("cutoff", ["2023-12-31", date(2024, 1, 1), "2026-01-01"])
def test_calendar_cutoff_rejects_empty_train_or_test_before_fitting(cutoff):
    def unexpected_fit(params):
        pytest.fail("Invalid date cutoff must fail before constructing a model")

    with pytest.raises(ValueError, match="nonempty training and test"):
        train_model(
            data=identified_data(),
            params={},
            model_factory=unexpected_fit,
            metric=mae,
            run_name="invalid-calendar-test",
            config=TrainingConfig(validation=HoldoutConfig(), test_size=cutoff),
        )


def test_latest_split_with_no_best_iteration_keeps_original_budget(tmp_path):
    setup_mlflow(tmp_path)

    class Model(MeanModel):
        def fit(self, train, validation=None):
            self._best = 3 if len(train) == 8 else None
            return super().fit(train, validation)

        @property
        def best_iteration(self):
            return self._best

    result = train_model(
        data=make_data(),
        params={"iterations": 30},
        model_factory=Model,
        metric=mae,
        run_name="last-budget",
        config=TrainingConfig(
            validation=TimeSeriesCVConfig(n_splits=2),
            tracking=TrackingConfig(log_final_model=False),
        ),
    )
    assert result.validation.splits[0].refit_params["iterations"] == 4
    assert result.params["iterations"] == 30
    assert result.test_metrics is None


def test_test_reuses_reports_and_evaluator_independently_of_split_selection(tmp_path):
    setup_mlflow(tmp_path)
    calls = []

    def evaluate(**inputs):
        calls.append((inputs["split"], inputs["object_id"].to_list()))
        return {"rows": float(len(inputs["object_id"]))}

    result = train_model(
        data=identified_data(),
        params={},
        model_factory=MeanModel,
        metric=mae,
        run_name="reports",
        post_fit_evaluator=CallbackEvaluator(name="business", evaluate=evaluate),
        config=TrainingConfig(
            validation=TimeSeriesCVConfig(n_splits=2),
            test_size=4,
            reporting=ReportingConfig(
                diagnostics=RegressionDiagnosticsConfig(
                    plots=RegressionPlotsConfig(
                        common=CommonPlotsConfig(
                            n_bins=3, numerical_distributions=None
                        ),
                        time_observed_fitted=False,
                    )
                ),
                validation_splits=(1,),
            ),
            tracking=TrackingConfig(log_final_model=False),
        ),
    )
    assert calls[-2:] == [
        (DatasetSplit.TRAIN, list(range(20))),
        (DatasetSplit.TEST, list(range(20, 24))),
    ]
    assert len(calls) == 6
    client = mlflow.MlflowClient()
    for run in mlflow.search_runs(output_format="list"):
        artifacts = list_all_artifacts(client, run.info.run_id)
        name = run.data.tags["mlflow.runName"]
        if name == "validation/split_1":
            assert "tables/psi/scores.json" in artifacts
        elif name == "validation/split_2":
            assert not artifacts
        elif name == "final_fit":
            assert "tables/psi/scores.json" in artifacts
            assert "figures/observed_fitted/by_feature/000_x.png" in artifacts
            assert "test/business/rows" in run.data.metrics
            assert "test/mae" in run.data.metrics
            assert "val/mae" not in run.data.metrics
            assert run.data.metrics["test/mae"] == result.test_metrics["mae"]


@pytest.mark.parametrize("fixed_threshold", [None, 0.8])
def test_supervised_final_fit_preserves_last_threshold(tmp_path, fixed_threshold):
    setup_mlflow(tmp_path)
    source = make_data()
    data = source.with_target(pl.Series("target", [0, 1] * 12))
    models = []

    def factory(params):
        model = CatBoostClassificationModel(params)
        models.append(model)
        return model

    params = {
        "iterations": 4,
        "thread_count": 1,
        "depth": 2,
        "verbose": False,
        "allow_writing_files": False,
    }
    if fixed_threshold is not None:
        params["decision_threshold"] = fixed_threshold
    result = train_model(
        data=data,
        params=params,
        model_factory=factory,
        metric=metrics_classification,
        run_name="threshold",
        config=TrainingConfig(
            validation=HoldoutConfig(val_size=0.25),
            tracking=TrackingConfig(log_final_model=False),
        ),
    )
    assert result.model.threshold == models[-2].threshold
    assert result.params["decision_threshold"] == models[-2].threshold
    assert result.model.native_model.tree_count_ == models[-2].best_iteration + 1
    assert result.model.validation_f1 is None


@pytest.mark.parametrize(
    "budget", ["iterations", "n_estimators", "num_trees", "num_boost_round"]
)
@pytest.mark.parametrize("best_iteration", [None, 4])
def test_refit_preserves_budget_alias_and_disables_early_stopping(
    budget, best_iteration
):
    params = {
        budget: 30,
        "early_stopping_rounds": 2,
        "od_type": "Iter",
        "od_wait": 2,
        "od_pval": 0.01,
        "use_best_model": True,
    }
    original = params.copy()
    refit = refit_parameters(params, best_iteration)
    assert refit == {
        budget: 30 if best_iteration is None else 5,
        "use_best_model": False,
    }
    assert params == original


def test_final_relearns_inflation_and_baseline_on_all_rows_before_test(tmp_path):
    setup_mlflow(tmp_path)
    data = make_data()
    models, inflation_fits = [], []
    reference_date = date(2024, 1, 1)

    class Inflation(ConstantRateInflation):
        def fit(self, dates, y, sample_weight=None):
            inflation_fits.append((dates.to_list(), y.copy(), sample_weight.copy()))
            return super().fit(dates, y, sample_weight)

    def factory(params):
        model = InflationAdjustedModel(
            CatBoostLogLinkRegressionModel(params),
            Inflation(0.03),
            reference_date,
        )
        models.append(model)
        return model

    result = train_model(
        data=data,
        params={
            "iterations": 4,
            "thread_count": 1,
            "depth": 2,
            "loss_function": "Poisson",
            "verbose": False,
            "allow_writing_files": False,
            "early_stopping_rounds": 2,
        },
        model_factory=factory,
        metric=mae,
        run_name="inflation-refit",
        config=TrainingConfig(
            validation=HoldoutConfig(val_size=4, gap=2),
            test_size=4,
            tracking=TrackingConfig(log_final_model=False),
        ),
    )
    assert [len(dates) for dates, _, _ in inflation_fits] == [14, 20]
    assert inflation_fits[-1][0] == data.time.head(20).to_list()
    np.testing.assert_array_equal(inflation_fits[-1][1], data.y.head(20).to_numpy())
    np.testing.assert_array_equal(
        inflation_fits[-1][2], data.weight.head(20).to_numpy()
    )
    normalized = result.model.inflation.adjust(
        data.y.head(20).to_numpy(),
        data.time.head(20),
        pl.Series([reference_date] * 20),
    )
    expected = np.log(np.average(normalized, weights=data.weight.head(20).to_numpy()))
    assert result.model.model.baseline == pytest.approx(expected)
    assert result.model.model.baseline != pytest.approx(models[0].model.baseline)
    assert result.model.model is not models[0].model
    assert result.params == result.validation.last_split.refit_params
    assert "early_stopping_rounds" not in result.params
    assert np.isfinite(result.model.predict_pool(data)).all()


def test_undefined_nonobjective_metric_is_preserved():
    from ml_toolkit.metrics import metrics_classification
    from ml_toolkit.training.evaluation import calculate_metrics

    data = make_data().with_target(pl.Series("target", [0.0] * 24))
    scores = calculate_metrics(metrics_classification, data, np.full(24, 0.2))
    assert np.isnan(scores["roc_auc"])
    assert np.isfinite(scores["log_loss"])


@pytest.mark.parametrize(
    "validation", [HoldoutConfig(val_size=8), TimeSeriesCVConfig(n_splits=2)]
)
@pytest.mark.parametrize("zero_window", ["train", "val"])
def test_zero_weight_windows_fail_before_any_fit(tmp_path, validation, zero_window):
    setup_mlflow(tmp_path)
    source = make_data()
    weights = np.ones(24)
    if zero_window == "train":
        weights[:16] = 0
    else:
        weights[8:] = 0
    data = Pool(source.X, source.y, source.time, pl.Series("weight", weights))
    fits = []

    class Model(MeanModel):
        def fit(self, train, validation=None):
            fits.append(train)
            return super().fit(train, validation)

    with pytest.raises(ValueError, match="positive total"):
        train_model(
            data=data,
            params={},
            model_factory=Model,
            metric=mae,
            run_name="invalid-window",
            config=TrainingConfig(validation=validation),
        )
    assert fits == []
    assert mlflow.search_runs(output_format="list") == []


def test_split_searches_are_independent_and_final_uses_last_post_tuning_fit(tmp_path):
    setup_mlflow(tmp_path)
    data = identified_data()
    events, ranked, packages = [], [], []

    class Model(MeanModel):
        def fit(self, train, validation=None):
            name = mlflow.active_run().data.tags["mlflow.runName"]
            events.append(
                (
                    name,
                    self.params["level"],
                    tuple(train.X.columns),
                    train.object_id.to_list(),
                    None if validation is None else validation.object_id.to_list(),
                )
            )
            super().fit(train, validation)
            trial = name.startswith("trial_")
            self._best = 99 if self.params["level"] == 0 else 2 if trial else len(train)
            self.threshold = (
                0.99
                if self.params["level"] == 0
                else 0.01 if trial else len(train) / 100
            )
            return self

        @property
        def best_iteration(self):
            return self._best

        def predict_pool(self, data):
            return np.full(len(data), self.params["level"], dtype=float)

        def get_refit_params(self):
            return {**super().get_refit_params(), "decision_threshold": self.threshold}

        def log(self, sample, *, config=None):
            packages.append(
                (
                    mlflow.active_run().data.tags["mlflow.runName"],
                    config.registered_model_name,
                )
            )
            super().log(sample, config=config)

    def ranker(model, validation):
        ranked.append(validation.object_id.to_list())
        early = validation.object_id[0] == 8
        return {"x": float(early), "unused": float(not early)}

    base = {
        "level": 0,
        "iterations": 100,
        "early_stopping_rounds": 2,
        "use_best_model": True,
    }
    result = train_model(
        data=data,
        params=base,
        model_factory=Model,
        metric=mae,
        param_space=lambda trial: {
            "level": (10, 18)[trial.number],
            "marker": trial.number,
        },
        run_name="independent-searches",
        config=TrainingConfig(
            validation=TimeSeriesCVConfig(n_splits=2),
            test_size=4,
            tuning=TuningConfig(metric="mae", direction="minimize", n_trials=2),
            tracking=TrackingConfig(
                log_trial_models=True,
                log_validation_models=True,
                registered_model_name="final-only",
            ),
        ),
        feature_selector=RFESelector(
            ranker=ranker, min_features=1, metric="mae", direction="minimize"
        ),
    )
    first, last = result.validation.splits
    assert [part.features for part in (first, last)] == [("x",), ("unused",)]
    assert [part.tuning.best_params["level"] for part in (first, last)] == [10, 18]
    assert [part.tuning.best_trial for part in (first, last)] == [0, 1]
    assert ranked == [list(range(8, 14)), list(range(14, 20))]
    for offset, part, train_end, val_end in [(0, first, 8, 14), (5, last, 14, 20)]:
        runs = events[offset : offset + 5]
        assert [event[1] for event in runs] == [
            0,
            0,
            10,
            18,
            part.tuning.best_params["level"],
        ]
        assert [event[2] for event in runs] == [("x", "unused")] + [part.features] * 4
        assert all(event[3] == list(range(train_end)) for event in runs)
        assert all(event[4] == list(range(train_end, val_end)) for event in runs)
    assert events[-1] == ("final_fit", 18, ("unused",), list(range(20)), None)
    assert result.tuning is last.tuning
    assert result.feature_selection is last.feature_selection
    assert result.params["iterations"] == 15
    assert result.params["decision_threshold"] == 0.14
    assert result.params["marker"] == 1
    assert result.params["use_best_model"] is False
    assert "early_stopping_rounds" not in result.params
    assert base == {
        "level": 0,
        "iterations": 100,
        "early_stopping_rounds": 2,
        "use_best_model": True,
    }
    assert result.validation.mean_metrics["mae"] == pytest.approx(
        np.mean([first.val_metrics["mae"], last.val_metrics["mae"]])
    )
    assert result.test_metrics == {"mae": 4.5}
    assert packages == [
        ("trial_1", None),
        ("trial_2", None),
        ("validation/split_1", None),
        ("trial_1", None),
        ("trial_2", None),
        ("validation/split_2", None),
        ("final_fit", "final-only"),
    ]
    runs = mlflow.search_runs(output_format="list")
    studies = [run for run in runs if run.data.tags["mlflow.runName"] == "tuning"]
    assert len(studies) == 2
    parents = {run.info.run_id: run.data.tags["mlflow.runName"] for run in runs}
    assert {parents[run.data.tags["mlflow.parentRunId"]] for run in studies} == {
        "validation/split_1",
        "validation/split_2",
    }
    for study in studies:
        trials = [
            run
            for run in runs
            if run.data.tags.get("mlflow.parentRunId") == study.info.run_id
        ]
        assert len(trials) == 2
        assert min(run.data.metrics["objective"] for run in trials) in [
            first.tuning.best_score,
            last.tuning.best_score,
        ]


@pytest.mark.parametrize("stage", ["fit", "metric"])
def test_training_failure_preserves_exception_and_callers_active_run(tmp_path, stage):
    setup_mlflow(tmp_path)
    original = RuntimeError("project callback failed")

    class Model(MeanModel):
        def fit(self, train, validation=None):
            if stage == "fit":
                raise original
            return super().fit(train, validation)

    def metric(y, prediction, weight):
        raise original

    with mlflow.start_run(run_name="caller") as caller:
        with pytest.raises(RuntimeError) as raised:
            train_model(
                data=make_data(),
                params={},
                model_factory=Model,
                metric=metric,
                run_name="failed-training",
                config=TrainingConfig(validation=HoldoutConfig()),
            )
        assert raised.value is original
        assert mlflow.active_run().info.run_id == caller.info.run_id
        children = [
            run for run in mlflow.search_runs(output_format="list")
            if run.info.run_id != caller.info.run_id
        ]
        assert children
        assert all(run.info.status == "FAILED" for run in children)
    assert mlflow.active_run() is None
