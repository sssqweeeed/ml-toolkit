from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import polars as pl
import pytest

from ml_toolkit.config import CommonPlotsConfig
from ml_toolkit.plots.monitoring import time_feature_distribution
from ml_toolkit.plots.regression import time_observed_fitted
from ml_toolkit.psi import PSIAnalysis, PSIConfig
from ml_toolkit.feature_levels import FeatureLevelConfig
from ml_toolkit._time import paired_period_starts, period_starts


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
            PSIConfig(features=(FeatureLevelConfig("x", "numerical"),), period=period)
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
                PSIConfig(features=(FeatureLevelConfig("x", "numerical"),), period="day")
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
