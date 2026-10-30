"""Calendar-window metrics preserve weights, sample boundaries and support."""

from dataclasses import replace
from datetime import date, datetime

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ml_toolkit.evaluation import MetricTrend
from ml_toolkit.config import FigureConfig, ReportingConfig
from ml_toolkit.data import Pool
from ml_toolkit.metrics import MetricScores, metrics_classification
from ml_toolkit.plots.metric_trend import metric_trend
from ml_toolkit.plots.utils import PlotError
from ml_toolkit.training.reporting import report_predictions
from tests.support.training import MeanModel


def mae(target, prediction, weight):
    return MetricScores(
        {"mae": float(np.average(abs(target - prediction), weights=weight))}
    )


def make_inputs(times, target=None, weight=None, split=None):
    length = len(times)
    return dict(
        data=Pool(
            X=pl.DataFrame({"x": np.arange(length, dtype=float)}),
            y=pl.Series(
                np.arange(1, length + 1, dtype=float) if target is None else target
            ),
            time=times if isinstance(times, pl.Series) else pl.Series(times),
            weight=pl.Series(np.ones(length) if weight is None else weight),
        ),
        prediction=np.zeros(length),
        split=pl.Series(["train"] * length if split is None else split),
    )


def test_monthly_metrics_are_weighted_and_calendar_windows_do_not_fill_gaps():
    inputs = make_inputs(
        [date(2024, 1, 2), date(2024, 1, 31), date(2024, 3, 1), date(2024, 4, 2)],
        target=[1.0, 5.0, 9.0, 13.0],
        weight=[1.0, 3.0, 2.0, 1.0],
    )
    report = MetricTrend(metric=mae, period="month", window=2)
    table = report.calculate(**inputs)
    assert table["period_start"].to_list() == [
        datetime(2024, 1, 1),
        datetime(2024, 3, 1),
        datetime(2024, 4, 1),
    ]
    assert table["window_start"].to_list() == [
        datetime(2023, 12, 1),
        datetime(2024, 2, 1),
        datetime(2024, 3, 1),
    ]
    assert table["n_rows"].to_list() == [2, 1, 2]
    assert table["sum_weight"].to_list() == [4.0, 2.0, 3.0]
    np.testing.assert_allclose(table["value"], [4.0, 9.0, 31 / 3])
    assert table["status"].to_list() == ["ok"] * 3
    assert report.parameters()["window"] == 2
    assert report.parameters()["metric"].endswith(".mae")


def test_interleaved_splits_and_rows_keep_callback_alignment_without_leakage():
    inputs = make_inputs(
        [date(2024, 3, 1), date(2024, 1, 1), date(2024, 3, 2), date(2024, 2, 1)],
        target=[100.0, 2.0, 200.0, 6.0],
        weight=[1.0, 2.0, 3.0, 4.0],
        split=["test", "train", "train", "test"],
    )
    seen = []

    def ordered(target, prediction, weight):
        seen.append((target.copy(), weight.copy()))
        return mae(target, prediction, weight)

    table = MetricTrend(metric=ordered, window=3).calculate(**inputs)
    train = table.filter(pl.col("split") == "train")
    test = table.filter(pl.col("split") == "test")
    np.testing.assert_allclose(train["value"], [2.0, 604 / 5])
    np.testing.assert_allclose(test["value"], [6.0, 124 / 5])
    # Windows retain original positional order, even with unsorted timestamps.
    np.testing.assert_array_equal(seen[1][0], [2.0, 200.0])
    np.testing.assert_array_equal(seen[-1][0], [100.0, 6.0])
    np.testing.assert_array_equal(seen[-1][1], [1.0, 4.0])


def test_supported_and_skipped_windows_keep_support_and_known_metric_names():
    inputs = make_inputs(
        [date(2024, 1, 1), date(2024, 2, 1), date(2024, 2, 2), date(2024, 3, 1)],
        weight=[0.0, 1.0, 2.0, 1.0],
    )
    calls = []

    def metric(target, prediction, weight):
        calls.append(len(target))
        return MetricScores({"finite": 2.0, "undefined": float("nan")})

    table = MetricTrend(metric=metric, min_rows=2).calculate(**inputs)
    assert calls == [2]
    assert table.height == 6
    january = table.filter(pl.col("period_start") == datetime(2024, 1, 1))
    assert january["status"].to_list() == ["zero_weight"] * 2
    assert january["n_rows"].to_list() == [1, 1]
    assert january["n_positive_weight_rows"].to_list() == [0, 0]
    march = table.filter(pl.col("period_start") == datetime(2024, 3, 1))
    assert march["status"].to_list() == ["insufficient_rows"] * 2
    undefined = table.filter(pl.col("status") == "undefined")
    assert undefined["metric"].to_list() == ["undefined"]
    assert undefined["value"].to_list() == [None]


def test_all_skipped_windows_do_not_invoke_metric_and_only_export_table(tmp_path):
    inputs = make_inputs([date(2024, 1, 1), date(2024, 2, 1)])

    def fail(*args):
        pytest.fail("Called metric for an undersupported window")

    report = MetricTrend(metric=fail, min_rows=3)
    table = report.calculate(**inputs)
    assert table["metric"].to_list() == [None, None]
    assert table["status"].to_list() == ["insufficient_rows"] * 2
    paths = report.write_artifacts(
        **inputs,
        artifact_dir=tmp_path,
        figure=FigureConfig(),
        features=None,
        n_bins=10,
    )
    assert [path.name for path in paths] == ["metrics.parquet"]
    assert_frame_equal(pl.read_parquet(paths[0]), table)


def test_single_class_metrics_have_undefined_values():
    inputs = make_inputs(
        [date(2024, 1, 1), date(2024, 2, 1)],
        target=[0.0, 1.0],
        split=["val", "val"],
    )
    table = MetricTrend(metric=metrics_classification).calculate(
        **{**inputs, "prediction": np.array([0.2, 0.8])}
    )
    roc = table.filter(pl.col("metric") == "roc_auc")
    assert roc["value"].to_list() == [None, None]
    assert roc["status"].to_list() == ["undefined", "undefined"]


@pytest.mark.parametrize(
    "period,times,starts",
    [
        (
            "week",
            [date(2024, 1, 7), date(2024, 1, 8)],
            [datetime(2024, 1, 1), datetime(2024, 1, 8)],
        ),
        (
            "quarter",
            [date(2024, 3, 31), date(2024, 4, 1)],
            [datetime(2024, 1, 1), datetime(2024, 4, 1)],
        ),
        (
            "year",
            [date(2023, 12, 31), date(2024, 1, 1)],
            [datetime(2023, 1, 1), datetime(2024, 1, 1)],
        ),
    ],
)
def test_calendar_boundaries(period, times, starts):
    table = MetricTrend(metric=mae, period=period, window=2).calculate(
        **make_inputs(times)
    )
    assert table["period_start"].to_list() == starts
    assert table["n_rows"].to_list() == [1, 2]


def test_timezone_and_daily_windows_preserve_dst_calendar_boundaries(tmp_path):
    times = pl.Series(
        [
            datetime(2024, 3, 30, 12),
            datetime(2024, 3, 31, 12),
            datetime(2024, 4, 1, 12),
        ]
    ).dt.replace_time_zone("Europe/Berlin")
    report = MetricTrend(metric=mae, period="day", window=2)
    table = report.calculate(**make_inputs(times))
    assert table.schema["period_start"] == pl.Datetime("us", "Europe/Berlin")
    assert table.schema["window_start"] == table.schema["period_start"]
    assert table["n_rows"].to_list() == [1, 2, 2]
    assert [value.hour for value in table["window_start"]] == [0, 0, 0]
    metric_trend(table, metric="mae", period="day", save_fig=tmp_path / "dst.png")
    assert (tmp_path / "dst.png").is_file()
    assert not plt.get_fignums()


def test_hour_windows_span_repeated_dst_hour_without_merging_buckets():
    times = (
        pl.Series(
            [
                datetime(2024, 10, 27, 0, 30),
                datetime(2024, 10, 27, 1, 30),
                datetime(2024, 10, 27, 2, 30),
            ]
        )
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone("Europe/Berlin")
    )
    table = MetricTrend(metric=mae, period="hour", window=2).calculate(
        **make_inputs(times)
    )
    assert table.height == 3
    assert table["n_rows"].to_list() == [1, 2, 2]
    assert [value.hour for value in table["period_start"]] == [2, 2, 3]


@pytest.mark.parametrize(
    "settings",
    [
        {"period": "fortnight"},
        {"window": 0},
        {"window": 1.5},
        {"window": True},
        {"min_rows": 0},
        {"min_rows": True},
    ],
)
def test_invalid_configuration(settings):
    with pytest.raises(ValueError):
        MetricTrend(metric=mae, **settings)


@pytest.mark.parametrize("period", ["minute", "hour"])
def test_date_inputs_reject_subday_periods(period):
    with pytest.raises(ValueError, match="requires Datetime"):
        MetricTrend(metric=mae, period=period).calculate(
            **make_inputs([date(2024, 1, 1)])
        )


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("prediction", np.array([float("nan"), 0.0]), "finite"),
        ("weight", pl.Series([1.0, -1.0]), "nonnegative"),
        ("weight", pl.Series([1.0, float("inf")]), "finite"),
        ("y", pl.Series([float("nan"), 1.0]), "finite"),
        ("split", pl.Series(["val", "test"]), "one held-out split"),
    ],
)
def test_input_validation(field, value, message):
    inputs = make_inputs([date(2024, 1, 1), date(2024, 2, 1)])
    if field in ("weight", "y"):
        inputs["data"] = replace(inputs["data"], **{field: value})
    else:
        inputs[field] = value
    with pytest.raises(ValueError, match=message):
        MetricTrend(metric=mae).calculate(**inputs)


def test_callback_errors_keep_original_exception_and_window_context():
    error = RuntimeError("domain failure")

    def fail(*args):
        raise error

    with pytest.raises(RuntimeError) as caught:
        MetricTrend(metric=fail, name="monthly").calculate(
            **make_inputs([date(2024, 1, 1)])
        )
    assert caught.value is error
    assert (
        "evaluator=monthly, split=train, period_start=2024-01-01" in error.__notes__[0]
    )


@pytest.mark.parametrize("bad", [{}, {"": 1.0}, {" ": 1.0}])
def test_bad_metric_names(bad):
    with pytest.raises(ValueError, match="nonempty named scores"):
        MetricTrend(metric=lambda *args: MetricScores(bad)).calculate(
            **make_inputs([date(2024, 1, 1)])
        )


def test_calculation_is_reusable_and_artifacts_compute_once(tmp_path, monkeypatch):
    inputs = make_inputs([date(2024, 1, 1), date(2024, 2, 1)])
    source = inputs["data"].X.clone()
    original = inputs["prediction"].copy()
    report = MetricTrend(metric=mae)
    expected = report.calculate(**inputs)
    assert_frame_equal(report.calculate(**inputs), expected)
    calls = []
    calculate = MetricTrend.calculate

    def counted(self, **kw):
        calls.append(1)
        return calculate(self, **kw)

    monkeypatch.setattr(MetricTrend, "calculate", counted)
    paths = report.write_artifacts(
        **inputs,
        artifact_dir=tmp_path,
        figure=FigureConfig(format="svg"),
        features=None,
        n_bins=10,
    )
    assert calls == [1]
    assert [path.name for path in paths] == ["metrics.parquet", "000_mae.svg"]
    assert all(path.is_file() for path in paths)
    assert_frame_equal(pl.read_parquet(paths[0]), expected)
    assert_frame_equal(inputs["data"].X, source)
    np.testing.assert_array_equal(inputs["prediction"], original)
    assert report.evaluate(**inputs) == {}
    assert not plt.get_fignums()


def test_plot_leaves_gaps_and_closes_on_save_failure(tmp_path, monkeypatch):
    table = MetricTrend(metric=mae).calculate(
        **make_inputs([date(2024, 1, 1), date(2024, 3, 1)])
    )
    lines = []

    def inspect(fig, *args, **kwargs):
        lines.append(fig.axes[0].lines[0].get_ydata())
        raise OSError("failed")

    monkeypatch.setattr("ml_toolkit.plots.metric_trend.save_figure", inspect)
    with pytest.raises(OSError, match="failed"):
        metric_trend(table, metric="mae", save_fig=tmp_path / "trend.png")
    assert len(lines[0]) == 3 and np.isnan(lines[0][1])
    assert not plt.get_fignums()
    with pytest.raises(PlotError, match="No metric trend"):
        metric_trend(table, metric="missing", save_fig=tmp_path / "bad.png")


def test_sparse_multi_year_minute_plot_uses_one_separator_per_gap(
    tmp_path, monkeypatch
):
    times = pl.Series(
        [
            datetime(2020, 1, 1, 12, 0),
            datetime(2025, 1, 1, 12, 0),
            datetime(2025, 1, 1, 12, 1),
        ]
    ).dt.replace_time_zone("Europe/Berlin")
    table = MetricTrend(metric=mae, period="minute").calculate(**make_inputs(times))
    shown = []

    def inspect(fig, *args, **kwargs):
        shown.extend(axis.lines[0].get_ydata() for axis in fig.axes)

    monkeypatch.setattr("ml_toolkit.plots.metric_trend.save_figure", inspect)
    metric_trend(table, metric="mae", period="minute", save_fig=tmp_path / "sparse.png")
    assert len(shown) == 2
    assert all(len(values) == 4 for values in shown)
    assert all(np.isnan(values[1]) for values in shown)
    assert not plt.get_fignums()


def test_report_uses_prediction_workflow_without_extra_scalar_metrics(monkeypatch):
    inputs = make_inputs([date(2024, 1, 1), date(2024, 2, 1)])
    artifacts, metrics = [], []
    monkeypatch.setattr(
        "ml_toolkit.training.reporting.tracking.log_artifact",
        lambda path, **kw: artifacts.append((path.name, kw)),
    )
    monkeypatch.setattr(
        "ml_toolkit.training.reporting.tracking.log_metrics",
        lambda values: metrics.append(values),
    )
    report_predictions(
        **inputs,
        model=MeanModel({}),
        evaluators=(MetricTrend(metric=mae),),
        config=ReportingConfig(),
        stage="final_fit",
    )
    assert [item[0] for item in artifacts] == ["metrics.parquet", "000_mae.png"]
    assert all(item[1]["artifact_path"] == "metric_trend" for item in artifacts)
    assert metrics == []
