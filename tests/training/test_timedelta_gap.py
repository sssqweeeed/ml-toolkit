"""Elapsed-time separation, independent of the density of observations."""

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from ml_toolkit.training import Holdout, TimeSeriesCV
from tests.support.training import make_identified_data


def timed_data(offsets, *, date=False, unit="us", zone=None):
    origin = datetime(2024, 1, 1, tzinfo=timezone.utc if zone else None)
    times = [origin + timedelta(hours=n) for n in offsets]
    series = pl.Series(
        "time", times, dtype=pl.Datetime(unit, time_zone="UTC" if zone else None)
    )
    if zone:
        series = series.dt.convert_time_zone(zone)
    if date:
        series = series.cast(pl.Date)
    return replace(make_identified_data().take(range(len(times))), time=series)


@pytest.mark.parametrize("unit", ["ms", "us", "ns"])
def test_irregular_gap_retains_exact_boundary_and_duplicates(unit):
    data = timed_data([0, 1, 4, 4, 5, 9, 10, 12], unit=unit)
    splitter = Holdout(data, val_size=2, gap=timedelta(hours=6))
    split = next(splitter.get_validation())
    assert split.train.object_id.to_list() == [0, 1, 2, 3]
    assert split.heldout.object_id.to_list() == [6, 7]
    assert splitter.get_test().train is data
    assert splitter.get_final() is data
    assert json.loads(json.dumps(splitter.parameters()))["gap"] == {
        "type": "timedelta",
        "microseconds": 6 * 3600 * 1_000_000,
    }


@pytest.mark.parametrize(
    "factory", [Holdout, lambda data, **kw: TimeSeriesCV(data, n_splits=2, **kw)]
)
def test_zero_duration_matches_zero_rows_with_ties(factory):
    data = timed_data([0, 0, 1, 1, 1, 1, 2, 2])
    for rows, duration in zip(
        factory(data, gap=0).get_validation(),
        factory(data, gap=timedelta()).get_validation(),
        strict=True,
    ):
        assert rows.train.object_id.equals(duration.train.object_id)
        assert rows.heldout.object_id.equals(duration.heldout.object_id)


def test_cv_windows_unchanged_and_max_train_size_applied_after_gap():
    data = timed_data([0, 1, 2, 3, 10, 11, 20, 21, 30, 31, 40, 41])
    splitter = TimeSeriesCV(
        data,
        n_splits=3,
        val_size=2,
        test_size=2,
        gap=timedelta(hours=9),
        max_train_size=2,
    )
    parts = list(splitter.get_validation())
    assert [p.train.object_id.to_list() for p in parts] == [[0, 1], [4, 5], [6, 7]]
    assert [p.heldout.object_id.to_list() for p in parts] == [[4, 5], [6, 7], [8, 9]]
    assert splitter.get_test().train.object_id.to_list() == list(range(10))
    assert splitter.get_test().heldout.object_id.to_list() == [10, 11]


def test_date_preserves_subday_gap():
    data = timed_data([0, 24, 48, 72, 96], date=True)
    split = next(Holdout(data, val_size=1, gap=timedelta(hours=25)).get_validation())
    assert split.train.object_id.to_list() == [0, 1, 2]


@pytest.mark.parametrize("month,day", [(3, 10), (11, 3)])
def test_dst_gap_is_elapsed_time(month, day):
    zone = ZoneInfo("America/New_York")
    start = datetime(2024, month, day, 0, tzinfo=zone).astimezone(timezone.utc)
    times = [start + timedelta(hours=n) for n in range(8)]
    data = replace(
        make_identified_data().take(range(8)),
        time=pl.Series(times).dt.convert_time_zone(str(zone)),
    )
    split = next(Holdout(data, val_size=2, gap=timedelta(hours=3)).get_validation())
    assert split.train.object_id.to_list() == [0, 1, 2, 3]


@pytest.mark.parametrize(
    "factory", [Holdout, lambda data, **kw: TimeSeriesCV(data, n_splits=2, **kw)]
)
@pytest.mark.parametrize("gap", [timedelta(microseconds=-1), timedelta(days=999999999)])
def test_invalid_or_empty_gap_fails_at_construction(factory, gap):
    with pytest.raises(ValueError, match="gap"):
        factory(timed_data(range(8)), gap=gap)


def test_skipped_empty_cv_window_does_not_block_later_windows():
    data = timed_data([0, 1, 2, 3, 10, 11, 20, 21])
    splitter = TimeSeriesCV(
        data, n_splits=3, val_size=2, skip_splits=1, gap=timedelta(hours=5)
    )
    assert [p.index for p in splitter.get_validation()] == [2, 3]


@pytest.mark.parametrize(
    "ticks,unit,gap,expected",
    [
        ([0, 1000, 2001, 2002, 3001], "ns", timedelta(microseconds=1), [0, 1, 2]),
        ([0, 1, 2, 3], "ms", timedelta(microseconds=1500), [0, 1]),
    ],
)
def test_gap_does_not_round_subunit_boundaries(ticks, unit, gap, expected):
    data = replace(
        make_identified_data().take(range(len(ticks))),
        time=pl.Series("time", ticks).cast(pl.Datetime(unit)),
    )
    split = next(Holdout(data, val_size=1, gap=gap).get_validation())
    assert split.train.object_id.to_list() == expected
