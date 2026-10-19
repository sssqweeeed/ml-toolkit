"""Behavioral contracts for selectors, weighted evaluation and temporal reports."""

from dataclasses import replace
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.metrics import f1_score, precision_score, recall_score

from examples.project_pool import ProjectPool
from examples.custom_selector import FixedFeatureSelector
from ml_toolkit import train_model
from ml_toolkit.config import (
    HoldoutConfig,
    TrainingConfig,
    TrackingConfig,
    CommonPlotsConfig,
)
from ml_toolkit.feature_selection import RFESelector, TopKSelector, ForwardSelector
from ml_toolkit.metrics import MetricScores
from ml_toolkit.models import CatBoostRegressionModel, CatBoostClassificationModel
from ml_toolkit.plots.classification import (
    binary_curves,
    calibration,
    precision_recall,
    f1_threshold,
)
from ml_toolkit.plots.monitoring import time_feature_distribution
from ml_toolkit.plots.regression import time_observed_fitted
from ml_toolkit.psi import PSIAnalysis, PSIConfig
from ml_toolkit.feature_levels import FeatureLevelSpec
from ml_toolkit._time import paired_period_starts, period_starts
from ml_toolkit.training import importance_ranker, split_holdout


def pool():
    return ProjectPool(
        X=pl.DataFrame({"a": range(12), "b": range(12), "c": range(12)}),
        y=pl.Series([0.0, 1.0] * 6),
        time=pl.Series([date(2025, 1, 1) + timedelta(days=i) for i in range(12)]),
        weight=pl.Series([1.0] * 12),
        numerical_features=("a", "b", "c"),
    )


def quality_metric(y, prediction, weight):
    return MetricScores({"score": float(np.average(prediction, weights=weight))})


def ranked(model, data):
    return {name: 1.0 for name in data.X.columns}


def quality_factory(quality, calls):
    class Model:
        best_iteration = None

        def __init__(self, params):
            self.params = params
            params["local"] = True

        def fit(self, train, validation=None):
            self.names = tuple(train.X.columns)
            calls.append((self, self.names, validation))
            return self

        def predict_pool(self, data):
            return np.full(len(data), quality[self.names])

        def get_refit_params(self):
            return {"budget": 1}

    return Model


def select(selector, quality):
    data = pool()
    part = split_holdout(data, HoldoutConfig())
    calls = []
    params = {"budget": 1}
    result = selector.select(
        train=part.train,
        validation=part.val,
        params=params,
        model_factory=quality_factory(quality, calls),
        metric=quality_metric,
    )
    assert params == {"budget": 1}
    assert len({id(model) for model, _, _ in calls}) == len(calls)
    assert result.steps[result.selected_step].features == result.features
    assert len(result.steps) == len(calls)
    return result, calls


def test_topk_keeps_exact_size_even_if_quality_deteriorates():
    result, calls = select(
        TopKSelector(k=2, ranker=ranked, protected_features=("c",)),
        {("a", "b", "c"): 1, ("a", "c"): 10},
    )
    assert result.features == ("a", "c")
    assert [step.metrics["score"] for step in result.steps] == [1, 10]
    assert len(calls) == 2
    all_result, calls = select(TopKSelector(k=3, ranker=ranked), {("a", "b", "c"): 1})
    assert len(calls) == 1
    assert all_result.settings["k"] == 3


@pytest.mark.parametrize("direction,sign", [("minimize", 1), ("maximize", -1)])
def test_forward_full_history_improvement_and_ties(direction, sign):
    quality = {
        ("a",): 3,
        ("b",): 3,
        ("c",): 4,
        ("a", "b"): 2,
        ("a", "c"): 2,
        ("a", "b", "c"): 2,
    }
    result, _ = select(
        ForwardSelector(
            metric="score", direction=direction, max_features=3, max_fits=6
        ),
        {key: sign * value for key, value in quality.items()},
    )
    assert result.features == ("a", "b")
    assert len(result.steps) == 6
    assert result.selected_step == 3
    assert result.steps[4].features == ("a", "c")
    assert result.steps[-1].features == ("a", "b", "c")


def test_forward_does_not_start_incomplete_round_and_preserves_protected():
    quality = {("b",): 4, ("a", "b"): 2, ("b", "c"): 1, ("a", "b", "c"): 0}
    result, _ = select(
        ForwardSelector(
            metric="score",
            direction="minimize",
            max_features=3,
            max_fits=3,
            protected_features=("b",),
        ),
        quality,
    )
    assert result.features == ("b", "c")
    assert len(result.steps) == 3
    result, _ = select(
        ForwardSelector(
            metric="score",
            direction="minimize",
            max_features=3,
            max_fits=3,
            min_improvement=3,
            protected_features=("b",),
        ),
        quality,
    )
    assert result.features == ("b",)
    with pytest.raises(ValueError, match="initial complete"):
        select(
            ForwardSelector(
                metric="score", direction="minimize", max_features=2, max_fits=2
            ),
            {},
        )
    result, _ = select(
        ForwardSelector(
            metric="score", direction="minimize", max_features=2, max_fits=4
        ),
        {("a",): 1, ("b",): 2, ("c",): 3},
    )
    assert result.features == ("a",)
    assert len(result.steps) == 3


@pytest.mark.parametrize(
    "selector",
    [
        RFESelector(
            min_features=1, metric="score", direction="minimize", ranker=ranked
        ),
        TopKSelector(k=1, ranker=ranked),
        ForwardSelector(
            metric="score", direction="minimize", max_features=2, max_fits=5
        ),
        FixedFeatureSelector(("b",)),
    ],
)
def test_selectors_integrate_with_fresh_final_fit(selector, tmp_path):
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/runs.db")
    mlflow.set_experiment("selectors")
    quality = {
        tuple(names): float(len(names))
        for names in (
            ("a",),
            ("b",),
            ("c",),
            ("a", "b"),
            ("a", "c"),
            ("b", "c"),
            ("a", "b", "c"),
        )
    }
    calls = []
    result = train_model(
        data=pool(),
        params={},
        model_factory=quality_factory(quality, calls),
        metric=quality_metric,
        run_name="selection",
        config=TrainingConfig(
            validation=HoldoutConfig(), tracking=TrackingConfig(log_final_model=False)
        ),
        feature_selector=selector,
    )
    assert calls[-1][1] == result.features
    assert calls[-1][2] is None
    assert result.feature_selection.settings
    assert len({id(model) for model, _, _ in calls}) == len(calls)


def test_weighted_curves_match_metrics_and_ignore_zero_rows():
    y = np.array([1.0, 0.0, 1.0, 0.0])
    p = np.array([0.9, 0.7, 0.4, 1.0])
    w = np.array([5.0, 20.0, 1.0, 0.0])
    curve = binary_curves(y, p, weight=w)
    for row in curve.itertuples():
        labels = p >= row.threshold
        assert row.f1 == pytest.approx(f1_score(y, labels, sample_weight=w))
        assert row.precision == pytest.approx(
            precision_score(y, labels, sample_weight=w)
        )
        assert row.recall == pytest.approx(recall_score(y, labels, sample_weight=w))
    threshold, score = CatBoostClassificationModel._select_threshold(y, p, w)
    assert threshold == curve.iloc[curve.f1.argmax()].threshold
    assert score == curve.f1.max()
    assert threshold != CatBoostClassificationModel._select_threshold(y[:3], p[:3])[0]
    pd.testing.assert_frame_equal(curve, binary_curves(y[:3], p[:3], weight=w[:3]))
    pd.testing.assert_frame_equal(
        binary_curves(y, p), binary_curves(y, p, weight=np.ones(4))
    )
    assert CatBoostClassificationModel._select_threshold(
        np.array([1.0, 0.0]), np.array([0.8, 0.2]), np.array([0.0, 1.0])
    ) == (0.5, 0.0)


def test_weighted_calibration_and_plots(tmp_path):
    from ml_toolkit.config import ClassificationPlotsConfig

    y, p, w = (
        np.array([0.0, 1.0, 1.0]),
        np.array([0.2, 0.8, 0.9]),
        np.array([3.0, 1.0, 0.0]),
    )
    table = calibration(
        y,
        p,
        tmp_path / "calibration.png",
        weight=w,
        config=ClassificationPlotsConfig(calibration_bins=1),
    )
    assert table.observed_rate.iloc[0] == 0.25
    assert table.predicted_rate.iloc[0] == pytest.approx(0.35)
    assert table.weight_sum.iloc[0] == 4
    for draw in (precision_recall, f1_threshold):
        result = draw(y, p, tmp_path / f"{draw.__name__}.png", weight=w)
        pd.testing.assert_frame_equal(result, binary_curves(y, p, weight=w))


@pytest.mark.parametrize("kind", [CatBoostRegressionModel, CatBoostClassificationModel])
def test_reserved_categories_checked_only_at_fit(kind):
    data = pool()
    data = replace(
        data,
        X=data.X.with_columns(pl.Series("cat", [None, "a"] * 6)),
        cat_features=("cat",),
    )
    params = {
        "iterations": 2,
        "thread_count": 1,
        "verbose": False,
        "allow_writing_files": False,
    }
    model = kind(params).fit(data)
    np.testing.assert_allclose(
        model.predict_pool(data), model.predict_frame(data.to_model_input())
    )
    invalid = replace(data, X=data.X.with_columns(pl.lit("unknown").alias("cat")))
    with pytest.raises(ValueError, match="cat.*reserved"):
        kind(params).fit(invalid)
    with pytest.raises(ValueError, match="cat.*reserved"):
        kind(params).fit(data, invalid)
    assert np.isfinite(model.predict_frame(invalid.to_model_input())).all()


@pytest.mark.parametrize(
    "period,expected",
    [
        ("minute", datetime(2025, 5, 14, 13, 27)),
        ("hour", datetime(2025, 5, 14, 13)),
        ("day", datetime(2025, 5, 14)),
        ("week", datetime(2025, 5, 12)),
        ("month", datetime(2025, 5, 1)),
        ("quarter", datetime(2025, 4, 1)),
        ("year", datetime(2025, 1, 1)),
    ],
)
def test_all_periods_agree_across_reports(period, expected, tmp_path):
    zone = ZoneInfo("Europe/Moscow")
    times = pl.Series([datetime(2025, 5, 14, 13, 27, 48, tzinfo=zone)])
    frame = pl.DataFrame({"x": [1.0], "time": times})
    current = frame.with_columns(pl.col("time") + pl.duration(seconds=1))
    result = (
        PSIAnalysis(
            PSIConfig(features=(FeatureLevelSpec("x", "numerical"),), period=period)
        )
        .fit(frame, time="time")
        .compare(current, time="time")
    )
    table = time_feature_distribution(
        frame, current, "x", "time", tmp_path / "dist.png", period=period
    )
    fitted = time_observed_fitted(
        times.to_list(),
        [1.0],
        [2.0],
        current["time"].to_list(),
        [1.0],
        [2.0],
        tmp_path / "fitted.png",
        period=period,
    )
    value = expected.replace(tzinfo=zone)
    assert result.time_distribution["period_start"][0] == value
    assert table["period_start"][0] == value
    assert fitted.period_start.iloc[0] == value
    assert table.schema["period_start"].time_zone == "Europe/Moscow"
    assert fitted.observed.tolist() == [1.0, 1.0]


def test_time_zone_boundaries_missing_periods_and_dst(tmp_path):
    zone = ZoneInfo("Europe/Berlin")
    # Both instances of the repeated DST hour must remain distinguishable.
    times = pl.Series(
        [
            datetime(2025, 10, 26, 2, 20, tzinfo=zone, fold=0),
            datetime(2025, 10, 26, 2, 40, tzinfo=zone, fold=1),
        ]
    )
    starts = period_starts(times, "hour")
    assert starts.n_unique() == 2
    assert (starts[1].timestamp() - starts[0].timestamp()) == 3600
    with pytest.raises(ValueError, match="matching time zones"):
        paired_period_starts(times, times.dt.convert_time_zone("UTC"), "day")
    with pytest.raises(ValueError, match="Datetime"):
        paired_period_starts(
            pl.Series([date(2025, 1, 1)]), pl.Series([date(2025, 1, 1)]), "hour"
        )
    frame = pl.DataFrame(
        {"x": [1.0, 3.0], "time": [datetime(2025, 1, 1), datetime(2025, 1, 3)]}
    )
    table = time_feature_distribution(
        frame,
        frame.with_columns(pl.col("time") + pl.duration(days=4)),
        "x",
        "time",
        tmp_path / "gaps.png",
        period="day",
    )
    assert table.height == 4
    assert table["period_start"].unique().sort().to_list() == [
        datetime(2025, 1, 1),
        datetime(2025, 1, 3),
        datetime(2025, 1, 5),
        datetime(2025, 1, 7),
    ]
    assert CommonPlotsConfig().period == "month"


def test_selector_history_snapshots_reused_metric_dictionary():
    data = pool()
    scores = {"score": 0.0}

    class Result:
        def to_dict(self):
            return scores

    def metric(y, prediction, weight):
        scores["score"] = float(prediction[0])
        return Result()

    result = RFESelector(
        min_features=1, max_fits=3, metric="score", direction="minimize", ranker=ranked
    ).select(
        train=data,
        validation=data,
        params={},
        model_factory=quality_factory(
            {("a", "b", "c"): 0, ("a", "b"): 10, ("a",): 20}, []
        ),
        metric=metric,
    )
    assert [step.metrics["score"] for step in result.steps] == [0, 10, 20]
    assert result.features == ("a", "b", "c")
    scores["score"] = 999
    assert result.steps[-1].metrics["score"] == 20


def test_weighted_plot_geometry_and_summary_labels(monkeypatch, tmp_path):
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score
    from ml_toolkit.plots import classification

    y, p, w = (
        np.array([1.0, 0.0, 1.0, 0.0]),
        np.array([0.9, 0.7, 0.4, 1.0]),
        np.array([5.0, 20.0, 1.0, 0.0]),
    )
    figures = []
    monkeypatch.setattr(
        classification, "save_figure", lambda fig, *a, **k: figures.append(fig)
    )
    try:
        precision_recall(y, p, tmp_path / "pr.png", weight=w)
        ax = figures[-1].axes[0]
        assert (
            ax.lines[0].get_label()
            == f"Model (weighted AP={average_precision_score(y, p, sample_weight=w):.3f})"
        )
        np.testing.assert_allclose(ax.lines[1].get_ydata(), np.average(y, weights=w))
        calibration(y, p, tmp_path / "cal.png", weight=w)
        histogram = figures[-1].axes[1]
        assert sum(patch.get_height() for patch in histogram.patches) == w.sum()
        assert "weight" in histogram.get_ylabel().lower()
    finally:
        for fig in figures:
            plt.close(fig)


@pytest.mark.parametrize(
    "offset",
    [
        timedelta(hours=3),
        timedelta(hours=5, minutes=30),
        timedelta(hours=-3, minutes=-30),
    ],
)
def test_time_observed_fitted_preserves_fixed_offsets(offset, tmp_path):
    from datetime import timezone

    zone = timezone(offset)
    times = [datetime(2025, 5, 14, 1, tzinfo=zone)]
    result = time_observed_fitted(
        times,
        [1.0],
        [1.0],
        [times[0] + timedelta(hours=1)],
        [1.0],
        [1.0],
        tmp_path / "fixed.png",
        period="day",
    )
    assert result.period_start.iloc[0] == datetime(2025, 5, 14, tzinfo=zone)
    assert result.period_start.iloc[0].utcoffset() == offset
    with pytest.raises(ValueError, match="matching time zones"):
        time_observed_fitted(
            times,
            [1.0],
            [1.0],
            [times[0].astimezone(timezone.utc)],
            [1.0],
            [1.0],
            tmp_path / "invalid.png",
        )


def test_time_plot_points_and_dst_bar_end(monkeypatch, tmp_path):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from ml_toolkit.plots import regression, psi

    figures = []
    monkeypatch.setattr(
        regression, "save_figure", lambda fig, *a, **k: figures.append(fig)
    )
    monkeypatch.setattr(psi, "save_figure", lambda fig, *a, **k: figures.append(fig))
    try:
        times = [datetime(2025, 1, 1, 12), datetime(2025, 1, 3, 18)]
        time_observed_fitted(
            times,
            [1.0, 3.0],
            [1.0, 2.0],
            [time + timedelta(days=4) for time in times],
            [1.0, 3.0],
            [1.0, 2.0],
            tmp_path / "points.png",
            period="day",
        )
        ax = figures[-1].axes[0]
        for line in ax.lines[1:]:
            offset = 4 if "current" in line.get_label() else 0
            assert list(pd.DatetimeIndex(line.get_xdata())) == [
                datetime(2025, 1, 1 + offset),
                datetime(2025, 1, 3 + offset),
            ]
        zone = ZoneInfo("America/Sao_Paulo")
        frame = pl.DataFrame(
            {
                "x": [1.0, 2.0],
                "time": [
                    datetime(2018, 11, 4, 12, tzinfo=zone),
                    datetime(2018, 11, 5, 12, tzinfo=zone),
                ],
            }
        )
        result = (
            PSIAnalysis(
                PSIConfig(features=(FeatureLevelSpec("x", "numerical"),), period="day")
            )
            .fit(frame.head(1), time="time")
            .compare(
                frame.with_columns(pl.col("time") + pl.duration(hours=1)), time="time"
            )
        )
        psi.time_psi_level_distribution(
            result.time_distribution, "x", tmp_path / "psi.png", period="day"
        )
        assert len(figures[-1].axes) == 1
        first, second, third = figures[-1].axes[0].patches[:3]
        assert first.get_x() == pytest.approx(
            mdates.date2num(datetime(2018, 11, 4, 1, tzinfo=zone))
        )
        assert first.get_x() + first.get_width() == pytest.approx(second.get_x())
        assert first.get_width() + second.get_width() == pytest.approx(23 / 24)
        assert second.get_x() + second.get_width() == pytest.approx(third.get_x())
    finally:
        for fig in figures:
            plt.close(fig)


def test_weighted_threshold_prefers_larger_threshold_on_exact_ties():
    y, p, w = (
        np.array([1.0, 0.0, 1.0]),
        np.array([0.9, 0.8, 0.7]),
        np.array([1.0, 2.0, 1.0]),
    )
    threshold, f1 = CatBoostClassificationModel._select_threshold(y, p, w)
    assert threshold == 0.9
    assert f1 == pytest.approx(2 / 3)
    tied = binary_curves(y, np.full(3, 0.6), weight=w)
    assert tied.threshold.to_list() == [0.6]
    assert tied.f1.to_list() == pytest.approx([2 / 3])


@pytest.mark.parametrize(
    "period,dates,expected",
    [
        (
            "week",
            [date(2025, 1, 1), date(2025, 1, 5), date(2025, 1, 6)],
            [datetime(2024, 12, 30), datetime(2024, 12, 30), datetime(2025, 1, 6)],
        ),
        (
            "quarter",
            [date(2024, 12, 31), date(2025, 1, 1), date(2025, 4, 1)],
            [datetime(2024, 10, 1), datetime(2025, 1, 1), datetime(2025, 4, 1)],
        ),
        (
            "year",
            [date(2024, 12, 31), date(2025, 1, 1)],
            [datetime(2024, 1, 1), datetime(2025, 1, 1)],
        ),
    ],
)
def test_calendar_boundaries(period, dates, expected):
    starts = period_starts(pl.Series(dates), period)
    assert starts.to_list() == expected
