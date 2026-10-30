"""Grouped metrics preserve typed keys and expose unsupported segment scores."""

from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ml_toolkit.config import FigureConfig, ReportingConfig
from ml_toolkit.data import Pool
from ml_toolkit.evaluation import GroupedMetrics, TimeObservedFitted
from ml_toolkit.metrics import MetricScores, metrics_classification, metrics_regression
from ml_toolkit.training.reporting import report_predictions
from tests.support.training import MeanModel


@pytest.fixture
def inputs():
    data = Pool(
        pl.DataFrame({"x": np.arange(8, dtype=float)}),
        pl.Series([0.0, 2.0, 1.0, 4.0, 3.0, 5.0, 6.0, 7.0]),
        pl.Series([date(2025, 1, 1) + timedelta(days=i) for i in range(8)]),
        pl.Series([1.0, 3.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0]),
        auxiliary_features=pl.DataFrame(
            {
                "segment": ["a", "a", "zero", None, "a", None, "zero", "test_only"],
                "region": [1, 1, 1, 1, 1, 2, 1, 1],
            }
        ),
    )
    return dict(
        data=data, prediction=np.ones(8), split=pl.Series(["train"] * 4 + ["test"] * 4)
    )


def test_weighted_scores_statuses_and_artifact_roundtrip(inputs, tmp_path):
    report = GroupedMetrics(metric=metrics_regression, by=("segment",), min_rows=2)
    table = report.calculate(**inputs)
    a = table.filter((pl.col("segment") == "a") & (pl.col("split") == "train"))
    assert a["sum_weight"].unique().to_list() == [4.0]
    assert a["n_positive_weight_rows"].unique().to_list() == [2]
    assert a.filter(pl.col("metric") == "mean_error")["value"].item() == -0.5
    assert table.filter(pl.col("segment") == "zero")["status"].unique().to_list() == [
        "zero_weight"
    ]
    assert set(table.filter(pl.col("segment") == "test_only")["status"]) == {
        "empty",
        "zero_weight",
    }
    assert table.filter(pl.col("segment").is_null())["status"].unique().to_list() == [
        "insufficient_rows"
    ]
    assert all(np.isfinite(value) for value in report.evaluate(**inputs).values())
    paths = report.write_artifacts(
        **inputs, artifact_dir=tmp_path, figure=FigureConfig(), features=None, n_bins=10
    )
    assert_frame_equal(pl.read_parquet(paths[0]), table)


def test_single_class_undefined_scores_and_train_only(inputs):
    data = replace(inputs["data"], y=pl.Series([0.0] * 8))
    report = GroupedMetrics(metric=metrics_classification, by=("segment",))
    args = {**inputs, "data": data, "prediction": np.full(8, 0.25)}
    table = report.calculate(**args)
    undefined = table.filter(pl.col("status") == "undefined")
    assert undefined.height > 0
    assert undefined["metric"].unique().to_list() == ["roc_auc"]
    assert undefined["value"].null_count() == undefined.height
    assert not any(key.endswith("/roc_auc") for key in report.evaluate(**args))
    train_only = {**args, "split": pl.Series(["train"] * 8)}
    assert report.calculate(**train_only)["split"].unique().to_list() == ["train"]


def test_all_skipped_groups_keep_null_metric_without_calling_metric(inputs):
    def fail(*args):
        raise AssertionError("Metric must not run")

    report = GroupedMetrics(metric=fail, by=("segment",), min_rows=100)
    table = report.calculate(**inputs)
    assert table["metric"].null_count() == table.height
    assert report.evaluate(**inputs) == {}


def test_metric_errors_propagate_and_unsafe_names_do_not_collide(inputs):
    def broken(*args):
        raise RuntimeError("Domain failure")

    with pytest.raises(RuntimeError, match="Domain failure"):
        GroupedMetrics(metric=broken, by=("segment",)).calculate(**inputs)

    def custom(*args):
        return MetricScores({"a/b": 1.0, "a_b": 2.0, "_hex_612f62": 3.0, "": 4.0})

    with pytest.raises(ValueError, match="nonempty"):
        GroupedMetrics(metric=custom, by=("segment",)).calculate(**inputs)

    def safe(*args):
        return MetricScores({"a/b": 1.0, "a_b": 2.0, "_hex_612f62": 3.0})

    result = GroupedMetrics(metric=safe, by=("segment",)).evaluate(**inputs)
    first = {
        key: value
        for key, value in result.items()
        if key.startswith("train/group_0000/")
    }
    assert len(first) == 3 and set(first.values()) == {1.0, 2.0, 3.0}


@pytest.mark.parametrize(
    "dtype,values",
    [
        (pl.String, [None, "None", "missing", "a/b", "a_b", "a/b", None, "missing"]),
        (pl.Float64, [None, float("nan"), 1.0, 2.0, 1.0, 2.0, None, 3.0]),
        (pl.Categorical, [None, "None", "a", "b", "a", "b", None, "c"]),
        (pl.Boolean, [True, False, None, True, False, None, True, False]),
    ],
)
def test_typed_missing_and_multicolumn_groups(inputs, dtype, values):
    data = replace(
        inputs["data"],
        auxiliary_features=inputs["data"].auxiliary_features.with_columns(
            pl.Series("segment", values, dtype=dtype)
        ),
    )
    report = GroupedMetrics(metric=metrics_regression, by=("segment", "region"))
    table = report.calculate(**{**inputs, "data": data})
    assert table.schema["segment"] == dtype
    keys = table.select("segment", "region", "group_id").unique()
    assert keys["group_id"].n_unique() == keys.height
    assert (
        table.filter(
            pl.col("segment").is_nan() if dtype == pl.Float64 else pl.lit(False)
        ).height
        == 0
    )


def test_time_groups_and_limits_use_exact_keys(inputs, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        "ml_toolkit.plots.regression.time_observed_fitted",
        lambda *a, **kw: calls.append(kw),
    )
    common = dict(
        **inputs, artifact_dir=tmp_path, figure=FigureConfig(), features=None, n_bins=10
    )
    TimeObservedFitted(by=("segment", "region")).write_artifacts(**common)
    labels = calls[0]["reference_group"].to_list() + calls[0]["current_group"].to_list()
    assert labels[0] == labels[1] == labels[4]
    assert labels[3] != labels[5]  # Missing segment with a different region.
    for report in (
        TimeObservedFitted(by=("x",), max_groups=2),
        GroupedMetrics(metric=metrics_regression, by=("x",), max_groups=2),
    ):
        with pytest.raises(ValueError, match="coarser groups"):
            report.write_artifacts(**common)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"by": ()},
        {"by": ("segment", "segment")},
        {"by": ("",)},
        {"by": ("metric",)},
        {"by": ("segment",), "min_rows": 0},
        {"by": ("segment",), "max_groups": 0},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        GroupedMetrics(metric=metrics_regression, **kwargs)


def test_training_runner_logs_only_finite_scores_and_table(inputs, monkeypatch):
    logged, artifacts = {}, []
    monkeypatch.setattr(
        "ml_toolkit.training.reporting.tracking.log_metrics",
        lambda values: logged.update(values),
    )
    monkeypatch.setattr(
        "ml_toolkit.training.reporting.tracking.log_artifact",
        lambda path, **kw: artifacts.append((path.name, kw)),
    )
    report = GroupedMetrics(metric=metrics_regression, by=("segment",))
    report_predictions(
        **inputs,
        model=MeanModel({}),
        evaluators=(report,),
        config=ReportingConfig(),
        stage="test"
    )
    assert logged and all(np.isfinite(value) for value in logged.values())
    assert artifacts == [("metrics.parquet", {"artifact_path": "grouped_metrics"})]
