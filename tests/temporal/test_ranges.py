"""Chronological sample boundaries in temporal diagnostics."""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import polars as pl
import pytest

from ml_toolkit._time import paired_period_starts
from ml_toolkit.feature_levels import FeatureLevelConfig
from ml_toolkit.plots import monitoring, psi, regression
from ml_toolkit.psi import PSIAnalysis, PSIConfig, compare_psi


@pytest.mark.parametrize(
    "reference,current",
    [
        # Ranges overlap even though the individual timestamps are distinct.
        ([1, 3], [2, 4]),
        ([3, 4], [1, 2]),
        ([1, 4], [2, 3]),
    ],
)
@pytest.mark.parametrize("report", ["psi", "observed_fitted", "feature"])
def test_temporal_reports_reject_overlapping_or_reversed_ranges(
    reference, current, report, tmp_path
):
    def sample(days):
        return pl.DataFrame(
            {"x": [1.0, 2.0], "time": [datetime(2025, 1, day) for day in days]}
        )

    before, after = sample(reference), sample(current)
    output = tmp_path / "invalid.png"
    with pytest.raises(ValueError, match=r"max\(reference\) <= min\(current\)"):
        if report == "psi":
            PSIAnalysis(PSIConfig(features=(FeatureLevelConfig("x", "numerical"),))).fit(
                before, time="time"
            ).compare(after, time="time")
        elif report == "observed_fitted":
            regression.time_observed_fitted(
                before["time"].to_list(),
                [1, 2],
                [1, 2],
                after["time"].to_list(),
                [1, 2],
                [1, 2],
                output,
            )
        else:
            monitoring.time_feature_distribution(before, after, "x", "time", output)
    assert not output.exists()


@pytest.mark.parametrize("kind", ["date", "datetime", "zoned"])
@pytest.mark.parametrize("single_day", [False, True])
def test_shared_endpoint_keeps_samples_separate_in_every_report(
    kind, single_day, monkeypatch, tmp_path
):
    def timestamp(day):
        if kind == "date":
            return date(2025, 1, day)
        zone = ZoneInfo("Europe/Moscow") if kind == "zoned" else None
        return datetime(2025, 1, day, tzinfo=zone)

    # Unsorted input; even samples entirely on the same date are valid.
    before = pl.DataFrame(
        {"x": [3.0, 1.0], "time": [timestamp(2), timestamp(2 if single_day else 1)]}
    )
    after = pl.DataFrame(
        {"x": [7.0, 5.0], "time": [timestamp(2 if single_day else 3), timestamp(2)]}
    )
    figures = []
    for module in (psi, monitoring, regression):
        monkeypatch.setattr(
            module, "save_figure", lambda fig, *a, **k: figures.append(fig)
        )
    try:
        result = (
            PSIAnalysis(
                PSIConfig(features=(FeatureLevelConfig("x", "numerical"),), period="day")
            )
            .fit(before, time="time")
            .compare(after, time="time")
        )
        table = monitoring.time_feature_distribution(
            before, after, "x", "time", tmp_path / "feature.png", period="day"
        )
        fitted = regression.time_observed_fitted(
            before["time"].to_list(),
            [3, 1],
            [3, 1],
            after["time"].to_list(),
            [7, 5],
            [7, 5],
            tmp_path / "fitted.png",
            period="day",
        )
        expected = [2.0, 6.0] if single_day else [1.0, 3.0, 5.0, 7.0]
        assert table["mean"].to_list() == expected
        assert fitted.observed.to_list() == expected
        assert fitted.fitted.to_list() == expected
        shares = result.time_distribution.group_by("sample", "period_start").agg(
            pl.col("share").sum()
        )
        assert shares.height == len(expected)
        assert shares["share"].to_list() == pytest.approx([1.0] * len(expected))
        psi.time_psi_level_distribution(
            result.time_distribution, "x", tmp_path / "psi.png", period="day"
        )
        ax = figures[-1].axes[0]
        # The two bars for the shared date touch, with each sample's own shares.
        bars = ax.patches[: len(expected)]
        split = len(expected) // 2
        left, right = bars[split - 1 : split + 1]
        boundary = ax.lines[0].get_xdata()[0]
        assert left.get_x() + left.get_width() == pytest.approx(boundary)
        assert right.get_x() == pytest.approx(boundary)
        assert all(bar.get_width() > 0 for bar in bars)
    finally:
        for fig in figures:
            plt.close(fig)


@pytest.mark.parametrize(
    "period", ["minute", "hour", "day", "week", "month", "quarter", "year"]
)
def test_split_within_one_period_is_allowed_by_every_report(
    period, monkeypatch, tmp_path
):
    figures = []
    for module in (psi, monitoring, regression):
        monkeypatch.setattr(
            module, "save_figure", lambda fig, *a, **k: figures.append(fig)
        )
    zone = ZoneInfo("Europe/Berlin")
    # Intentionally unsorted, with both samples falling in the same minute.
    before = pl.DataFrame(
        {
            "x": [1.0, 3.0],
            "time": [
                datetime(2025, 5, 14, 13, 27, 20, tzinfo=zone),
                datetime(2025, 5, 14, 13, 27, 10, tzinfo=zone),
            ],
        }
    )
    after = before.with_columns(pl.col("time") + pl.duration(seconds=30))
    try:
        comparison = (
            PSIAnalysis(
                PSIConfig(features=(FeatureLevelConfig("x", "numerical"),), period=period)
            )
            .fit(before, time="time")
            .compare(after, time="time")
        )
        table = monitoring.time_feature_distribution(
            before, after, "x", "time", tmp_path / "feature.png", period=period
        )
        fitted = regression.time_observed_fitted(
            before["time"].to_list(),
            [1, 3],
            [2, 2],
            after["time"].to_list(),
            [1, 3],
            [2, 2],
            tmp_path / "fitted.png",
            period=period,
        )
        psi.time_psi_level_distribution(
            comparison.time_distribution, "x", tmp_path / "psi.png", period=period
        )
        assert table["period_start"].n_unique() == 1
        assert fitted.period_start.nunique() == 1
        assert table["mean"].to_list() == [2.0, 2.0]
        assert fitted.observed.to_list() == [2.0, 2.0]
        assert (
            comparison.time_distribution["period_start"].unique().to_list()
            == table["period_start"].unique().to_list()
        )
        ax = figures[-1].axes[0]
        boundary = ax.lines[0].get_xdata()[0]
        for bar in ax.patches[::2]:
            assert bar.get_x() + bar.get_width() == pytest.approx(boundary)
        for bar in ax.patches[1::2]:
            assert bar.get_x() == pytest.approx(boundary)
        assert {text.get_text() for text in ax.texts} == {"REFERENCE", "CURRENT"}
    finally:
        for fig in figures:
            plt.close(fig)


def test_range_check_uses_original_instants_across_dst():
    zone = ZoneInfo("Europe/Berlin")
    reference = pl.Series([datetime(2025, 10, 26, 2, 50, tzinfo=zone, fold=0)])
    current = pl.Series([datetime(2025, 10, 26, 2, 10, tzinfo=zone, fold=1)])
    first, second = paired_period_starts(reference, current, "day")
    assert first.equals(second)
    with pytest.raises(ValueError, match="must precede or equal"):
        paired_period_starts(current, reference, "day")


def test_range_check_preserves_nanosecond_order():
    reference = pl.Series([1_735_689_600_000_000_001], dtype=pl.Datetime("ns"))
    current = reference + pl.duration(nanoseconds=1)
    # Expression evaluation preserves the source precision.
    current = pl.select(current).to_series()
    first, second = paired_period_starts(reference, current, "minute")
    assert first.equals(second)
    with pytest.raises(ValueError, match="must precede or equal"):
        paired_period_starts(current, reference, "minute")


def test_compare_psi_validates_times_only_when_requested():
    frame = pl.DataFrame({"x": [1.0, 2.0]})
    analysis = PSIAnalysis(
        PSIConfig(features=(FeatureLevelConfig("x", "numerical"),))
    ).fit(frame)
    assert analysis.compare(frame).psi["psi"][0] == 0
    config = analysis.transformer.feature_levels
    assert config is not None
    levels = analysis.transformer.transform(frame)
    reference = pl.Series([datetime(2025, 1, 1), datetime(2025, 1, 3)])
    with pytest.raises(ValueError, match="must precede or equal"):
        compare_psi(
            levels,
            levels,
            config,
            reference_time=reference,
            current_time=reference + timedelta(days=1),
        )


def test_psi_plot_rejects_overlapping_aggregated_ranges(tmp_path):
    distribution = pl.DataFrame(
        {
            "feature": ["x"] * 4,
            "period_start": [datetime(2025, month, 1) for month in [1, 3, 2, 4]],
            "sample": ["reference", "reference", "current", "current"],
            "level": ["low"] * 4,
            "share": [1.0] * 4,
        }
    )
    with pytest.raises(ValueError, match="only the boundary period may be shared"):
        psi.time_psi_level_distribution(distribution, "x", tmp_path / "invalid.png")
    assert not (tmp_path / "invalid.png").exists()
