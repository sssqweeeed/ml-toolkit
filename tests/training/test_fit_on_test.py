"""Main and optional full-data models have distinct lifetimes and scores."""

from dataclasses import replace
from itertools import product

import mlflow
import polars as pl
import pytest

from ml_toolkit.config import ReportingConfig, TrackingConfig, TrainingConfig
from ml_toolkit.training import Holdout, Split, train_model
from tests.support.evaluators import CallbackEvaluator
from tests.support.training import MeanModel, mae, make_identified_data, setup_mlflow


@pytest.mark.parametrize("has_test,fit_on_test", list(product((False, True), repeat=2)))
def test_fit_matrix_rows_models_scores_and_runs(tmp_path, has_test, fit_on_test):
    setup_mlflow(tmp_path)
    data = make_identified_data()
    events, reports, instances, full_requests, test_requests = [], [], [], [], []

    class Splitter(Holdout):
        def get_test(self):
            assert [event[0] for event in events] == ["validation/split_1"]
            test_requests.append(1)
            return super().get_test()

        def get_final(self):
            full_requests.append(1)
            assert "test" in [event[0] for event in events]
            return super().get_final()

    class Model(MeanModel):
        def fit(self, train, validation=None):
            instances.append(self)
            events.append(
                (
                    mlflow.active_run().data.tags["mlflow.runName"],
                    train.object_id.to_list(),
                    validation is None,
                )
            )
            return super().fit(train, validation)

        @property
        def best_iteration(self):
            return 2

        def get_refit_params(self):
            return {**super().get_refit_params(), "decision_threshold": 0.3}

    def record(**inputs):
        reports.append(
            (
                mlflow.active_run().data.tags["mlflow.runName"],
                inputs["split"].unique(maintain_order=True).to_list(),
            )
        )
        return {}

    splitter = Splitter(data, val_size=4, gap=2, test_size=4 if has_test else None)
    kwargs = dict(
        splitter=splitter,
        params={"iterations": 8, "early_stopping_rounds": 2, "use_best_model": True},
        model_factory=Model,
        metric=mae,
        run_name="matrix",
        config=TrainingConfig(
            fit_on_test=fit_on_test,
            tracking=TrackingConfig(log_test_model=False, log_final_model=False),
        ),
        prediction_evaluators=CallbackEvaluator(name="record", evaluate=record),
    )
    if fit_on_test and not has_test:
        with pytest.raises(ValueError, match="fit_on_test.*test"):
            train_model(**kwargs)
        assert [event[0] for event in events] == ["validation/split_1"]
        assert [report[0] for report in reports] == ["validation/split_1"]
        assert full_requests == []
        assert not any(
            run.data.tags["mlflow.runName"] in ("test", "final_fit")
            for run in mlflow.search_runs(output_format="list")
        )
        assert test_requests == [1]
        return

    result = train_model(**kwargs)
    end = 20 if has_test else 24
    assert test_requests == [1]
    assert events[0] == ("validation/split_1", list(range(end - 6)), False)
    main_stage = "test" if has_test else "final_fit"
    assert events[1] == (main_stage, list(range(end)), True)
    assert result.model is instances[1]
    assert result.model.mean == pytest.approx((end + 1) / 2)
    assert result.params == {
        "iterations": 3,
        "use_best_model": False,
        "decision_threshold": 0.3,
    }
    assert result.model.params == result.params
    assert result.test_metrics == ({"mae": 12.0} if has_test else None)
    assert reports[1] == (main_stage, ["train", "test"] if has_test else ["train"])
    if fit_on_test:
        assert events[2] == ("final_fit", list(range(24)), True)
        assert (
            result.final_model is instances[2]
            and result.final_model is not result.model
        )
        assert result.final_model.mean == pytest.approx(12.5)
        assert result.final_model.params == result.params
        assert reports[2] == ("final_fit", ["train"])
        assert full_requests == [1]
    else:
        assert len(events) == 2 and result.final_model is None and full_requests == []
    runs = {
        run.data.tags["mlflow.runName"]: run
        for run in mlflow.search_runs(output_format="list")
    }
    assert runs[main_stage].data.params["train_n_rows"] == str(end)
    if "final_fit" in runs:
        assert not any(
            key.startswith("test/") for key in runs["final_fit"].data.metrics
        )
        assert runs["final_fit"].data.params["train_n_rows"] == "24"
        assert "train/mae" in runs["final_fit"].data.metrics


@pytest.mark.parametrize(
    "fit_on_test,log_test,log_final", list(product((False, True), repeat=3))
)
def test_model_logging_and_registration_no_fallback(
    tmp_path, fit_on_test, log_test, log_final
):
    setup_mlflow(tmp_path)
    saved = []

    class Model(MeanModel):
        def log(self, sample, *, config=None, extra_artifacts=None):
            saved.append(
                (
                    mlflow.active_run().data.tags["mlflow.runName"],
                    config.registered_model_name,
                    len(sample),
                )
            )

    train_model(
        splitter=Holdout(make_identified_data(), test_size=4),
        params={},
        model_factory=Model,
        metric=mae,
        run_name="logging",
        config=TrainingConfig(
            fit_on_test=fit_on_test,
            tracking=TrackingConfig(
                log_test_model=log_test,
                log_final_model=log_final,
                registered_model_name="chosen",
            ),
        ),
    )
    expected = []
    if log_test:
        expected.append(("test", None if fit_on_test else "chosen", 20))
    if fit_on_test and log_final:
        expected.append(("final_fit", "chosen", 24))
    assert saved == expected


@pytest.mark.parametrize("log_test,log_final", [(True, False), (False, True)])
def test_no_test_uses_final_logging_flag(tmp_path, log_test, log_final):
    setup_mlflow(tmp_path)
    saved = []

    class Model(MeanModel):
        def log(self, sample, *, config=None, extra_artifacts=None):
            saved.append((len(sample), config.registered_model_name))

    result = train_model(
        splitter=Holdout(make_identified_data()),
        params={},
        model_factory=Model,
        metric=mae,
        run_name="no-test",
        config=TrainingConfig(
            tracking=TrackingConfig(
                log_test_model=log_test,
                log_final_model=log_final,
                registered_model_name="main",
            )
        ),
    )
    assert saved == ([(24, "main")] if log_final else [])
    assert result.final_model is None


@pytest.mark.parametrize("on_test,on_final_fit", list(product((False, True), repeat=2)))
def test_reporting_flags_are_independent(tmp_path, on_test, on_final_fit):
    setup_mlflow(tmp_path)
    stages = []

    def evaluate(**inputs):
        stages.append(mlflow.active_run().data.tags["mlflow.runName"])
        return {}

    result = train_model(
        splitter=Holdout(make_identified_data(), test_size=4),
        params={},
        model_factory=MeanModel,
        metric=mae,
        run_name="flags",
        config=TrainingConfig(
            fit_on_test=True,
            reporting=ReportingConfig(
                on_validation=False, on_test=on_test, on_final_fit=on_final_fit
            ),
            tracking=TrackingConfig(log_test_model=False, log_final_model=False),
        ),
        prediction_evaluators=CallbackEvaluator(name="record", evaluate=evaluate),
    )
    assert stages == (["test"] if on_test else []) + (
        ["final_fit"] if on_final_fit else []
    )
    assert result.test_metrics == {"mae": 12.0}
    runs = {
        r.data.tags["mlflow.runName"]: r
        for r in mlflow.search_runs(output_format="list")
    }
    assert "train/mae" in runs["final_fit"].data.metrics


def test_custom_full_pool_respects_policy_and_preserves_metadata(tmp_path):
    setup_mlflow(tmp_path)
    data = make_identified_data()
    full = data.take(range(2, 24))
    full_fits = []

    class Custom(Holdout):
        def get_test(self):
            return Split(self.data.take(range(2, 20)), self.data.take(range(20, 24)))

        def get_final(self):
            return full

    class Model(MeanModel):
        def fit(self, train, validation=None):
            if mlflow.active_run().data.tags["mlflow.runName"] == "final_fit":
                full_fits.append(train)
            return super().fit(train, validation)

    result = train_model(
        splitter=Custom(data, test_size=4),
        params={},
        model_factory=Model,
        metric=mae,
        run_name="custom",
        config=TrainingConfig(
            fit_on_test=True,
            tracking=TrackingConfig(log_test_model=False, log_final_model=False),
        ),
    )
    assert full_fits[0].object_id.equals(full.object_id)
    assert full_fits[0].business_type.equals(full.business_type)
    assert result.model.mean == pytest.approx(11.5)
    assert result.final_model.mean == pytest.approx(13.5)
    assert result.test_metrics == {"mae": 11.0}


@pytest.mark.parametrize("invalid", ["schema", "time"])
def test_invalid_full_pool_fails_before_additional_fit(tmp_path, invalid):
    setup_mlflow(tmp_path)
    fits = []

    class Custom(Holdout):
        def get_final(self):
            if invalid == "schema":
                return replace(
                    self.data,
                    X=self.data.X.with_columns(self.data.X["x"].cast(pl.Int64)),
                )
            return replace(self.data, time=self.data.time.reverse())

    class Model(MeanModel):
        def fit(self, train, validation=None):
            fits.append(len(train))
            return super().fit(train, validation)

    with pytest.raises(ValueError, match="schema|ascending"):
        train_model(
            splitter=Custom(make_identified_data(), test_size=4),
            params={},
            model_factory=Model,
            metric=mae,
            run_name="invalid-full",
            config=TrainingConfig(
                fit_on_test=True,
                tracking=TrackingConfig(log_test_model=False, log_final_model=False),
            ),
        )
    assert len(fits) == 2
