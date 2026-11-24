"""Temporal split membership, repeatability, metadata and boundary validation."""

from dataclasses import replace
from datetime import date, datetime, timedelta
from math import ceil

import numpy as np
import polars as pl
import pytest
from sklearn.model_selection import TimeSeriesSplit

from ml_toolkit.training import Holdout, Split, TimeSeriesCV
from tests.support.training import make_identified_data as identified_data


@pytest.mark.parametrize("size, count", [(3, 3), (0.2, 3), (0.01, 1), (1, 1)])
def test_tail_sizes_use_rows_or_ceiling_fraction(size, count):
    data = identified_data().take(range(11))
    split = next(Holdout(data, val_size=size).get_validation())
    assert type(split.train) is type(data) is type(split.heldout)
    assert split.train.object_id.to_list() == list(range(11 - count))
    assert split.heldout.object_id.to_list() == list(range(11 - count, 11))


def test_holdout_gap():
    data = identified_data().take(range(11))
    split = next(Holdout(data, val_size=0.25, gap=2).get_validation())
    assert split.train.object_id.to_list() == list(range(6))
    assert split.heldout.object_id.to_list() == [8, 9, 10]


def test_test_and_validation_fractions_use_their_current_input_and_round_up():
    data = identified_data().take(range(11))
    splitter = Holdout(data, test_size=0.2, val_size=0.25, gap=1)
    split = next(splitter.get_validation())
    final = splitter.get_test()
    assert (len(final.train), len(final.heldout)) == (8, 3)
    assert (len(split.train), len(split.heldout)) == (5, 2)
    assert final.train.object_id.to_list() == list(range(8))
    assert final.heldout.object_id.to_list() == [8, 9, 10]


@pytest.mark.parametrize("size", [0, -1, 0.0, 1.0, float("nan"), float("inf")])
def test_invalid_sizes_fail_at_construction(size):
    data = identified_data()
    for factory in (
        Holdout,
        lambda data, **kwargs: TimeSeriesCV(data, n_splits=2, **kwargs),
    ):
        with pytest.raises(ValueError):
            factory(data, test_size=size)
        with pytest.raises(ValueError):
            factory(data, val_size=size)


def test_cv_fraction_window_gap_and_original_indices():
    data = identified_data()
    splitter = TimeSeriesCV(
        data,
        test_size=4,
        n_splits=3,
        val_size=0.2,
        gap=2,
        max_train_size=5,
        skip_splits=1,
    )
    splits = list(splitter.get_validation())
    assert [split.index for split in splits] == [2, 3]
    for split in splits:
        assert len(split.train) == 5 and len(split.heldout) == 4
        assert split.heldout.object_id[0] - split.train.object_id[-1] == 3
    assert splitter.get_test().train.object_id.to_list() == list(range(20))


@pytest.mark.parametrize("kwargs", [{"val_size": 5}, {"val_size": 0.8, "gap": 2}])
def test_empty_train_is_rejected(kwargs):
    with pytest.raises(ValueError, match="Not enough rows"):
        Holdout(identified_data().take(range(5)), **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_splits": 1},
        {"n_splits": 3, "skip_splits": 3},
        {"n_splits": 3, "skip_splits": -1},
        {"n_splits": 3, "gap": -1},
        {"n_splits": 3, "max_train_size": 0},
        {"n_splits": 30},
        {"n_splits": 3, "gap": 20},
        {"n_splits": 3, "val_size": 8},
    ],
)
def test_invalid_cv_windows(kwargs):
    with pytest.raises(ValueError):
        TimeSeriesCV(identified_data(), **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_splits": 2},
        {"n_splits": 3, "gap": 1, "max_train_size": 4},
        {"n_splits": 3, "val_size": 3, "skip_splits": 1},
    ],
)
def test_cv_matches_sklearn_and_returns_fresh_iterators(kwargs):
    data = identified_data()
    splitter = TimeSeriesCV(data, test_size=4, **kwargs)
    expected = list(
        TimeSeriesSplit(
            n_splits=kwargs["n_splits"],
            test_size=kwargs.get("val_size"),
            gap=kwargs.get("gap", 0),
            max_train_size=kwargs.get("max_train_size"),
        ).split(range(20))
    )
    expected = expected[kwargs.get("skip_splits", 0) :]
    first = splitter.get_validation()
    second = splitter.get_validation()
    assert first is not second
    for iterator in (first, second):
        for actual, (train, val) in zip(iterator, expected, strict=True):
            np.testing.assert_array_equal(actual.train.object_id, train)
            np.testing.assert_array_equal(actual.heldout.object_id, val)


@pytest.mark.parametrize(
    "factory", [Holdout, lambda data: TimeSeriesCV(data, n_splits=3)]
)
def test_train_only_final_and_repeated_ids_keep_metadata(factory):
    source = identified_data()
    data = replace(
        source, object_id=pl.Series("id", [i // 2 for i in range(len(source))])
    )
    splitter = factory(data)
    assert splitter.get_test().train is data
    assert splitter.get_test().heldout is None
    for split in splitter.get_validation():
        for part in (split.train, split.heldout):
            indices = part.X["x"].cast(pl.Int64)
            for name in ("y", "time", "weight", "auxiliary_features", "object_id"):
                assert getattr(part, name).equals(getattr(data, name)[indices])
            assert part.cat_features == data.cat_features
            assert part.numerical_features == data.numerical_features
            assert type(part) is type(data)


@pytest.mark.parametrize(
    "factory",
    [Holdout, lambda data, **kwargs: TimeSeriesCV(data, n_splits=2, **kwargs)],
)
@pytest.mark.parametrize("cutoff", ["2024-02-29", date(2024, 2, 29)])
def test_test_date_is_normalized_and_serializable(factory, cutoff):
    data = identified_data()
    data = replace(
        data,
        time=pl.Series(
            "time", [date(2024, 2, 1) + timedelta(days=2 * i) for i in range(len(data))]
        ),
    )
    splitter = factory(data, test_size=cutoff)
    assert splitter.test_size == date(2024, 2, 29)
    assert splitter.parameters()["test_size"] == "2024-02-29"
    assert splitter.get_test().heldout.time.min() == date(2024, 2, 29)


@pytest.mark.parametrize(
    "cutoff",
    [
        "",
        "invalid",
        "2025-02-29",
        "2025-13-01",
        "2025-01-00",
        "20250101",
        "2025-W01-1",
        "2025-1-1",
        " 2025-01-01",
        "2025-01-01 ",
        "2025-01-01T12:00:00",
    ],
)
def test_invalid_test_date(cutoff):
    with pytest.raises(ValueError, match="test_size date must use YYYY-MM-DD"):
        Holdout(identified_data(), test_size=cutoff)


def test_exact_test_time_is_rejected():
    with pytest.raises(ValueError, match="calendar date without a time"):
        Holdout(identified_data(), test_size=datetime(2025, 1, 1))


@pytest.mark.parametrize("field", ["columns", "dtype", "categorical", "numerical"])
def test_pair_rejects_incompatible_features(field):
    data = identified_data()
    heldout = data.take(range(12, 24))
    if field == "columns":
        heldout = heldout.select_features(["x"])
    elif field == "dtype":
        heldout = replace(
            heldout, X=heldout.X.with_columns(pl.col("x").cast(pl.Float32))
        )
    elif field == "categorical":
        heldout = replace(heldout, cat_features=("x",), numerical_features=("unused",))
    else:
        heldout = replace(heldout, numerical_features=())
    with pytest.raises(ValueError, match="matching features"):
        Split(data.take(range(12)), heldout, index=1)


def panel_data(dates=8, per_date=3):
    data = identified_data().take(range(dates * per_date))
    time = pl.Series(
        "date",
        [date(2024, 1, 1) + timedelta(days=n // per_date) for n in range(len(data))],
    )
    return replace(data, time=time)


def assert_no_shared_timestamp(split):
    assert split.train.time.max() < split.heldout.time.min()
    assert not set(split.train.time) & set(split.heldout.time)


def test_holdout_fractions_never_split_a_timestamp():
    splitter = Holdout(panel_data(), val_size=0.2, test_size=0.2)
    final = splitter.get_test()
    assert_no_shared_timestamp(final)
    assert len(final.heldout) == 6
    split = next(splitter.get_validation())
    assert_no_shared_timestamp(split)
    assert split.heldout.object_id.to_list() == list(range(12, 18))
    assert split.train.object_id.to_list() == list(range(12))


def test_holdout_zero_timedelta_gap_with_integer_val_size():
    split = next(Holdout(panel_data(), val_size=4, gap=timedelta(0)).get_validation())
    assert_no_shared_timestamp(split)
    assert split.heldout.object_id.to_list() == list(range(18, 24))
    assert split.train.object_id.to_list() == list(range(18))


def test_holdout_integer_gap_drops_rows_before_snapped_start():
    split = next(Holdout(panel_data(), val_size=4, gap=2).get_validation())
    assert split.heldout.object_id.to_list() == list(range(18, 24))
    assert split.train.object_id.to_list() == list(range(16))


@pytest.mark.parametrize("gap", [0, 3, timedelta(0), timedelta(days=1)])
def test_cv_panel_windows_never_split_a_timestamp(gap):
    data = panel_data()
    windows = list(TimeSeriesCV(data, n_splits=3, val_size=4, gap=gap).get_validation())
    assert [split.index for split in windows] == [1, 2, 3]
    held = [row for split in windows for row in split.heldout.object_id.to_list()]
    assert held == list(range(12, 24))
    for split in windows:
        assert_no_shared_timestamp(split)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"val_size": 3},
        {"val_size": 0.1},
        {"test_size": 4, "val_size": 3},
        {"gap": 2, "max_train_size": 5},
        {"skip_splits": 1},
    ],
)
def test_cv_unique_timestamps_match_sklearn(kwargs):
    data = identified_data()
    skip = kwargs.get("skip_splits", 0)
    end = len(data) - kwargs.get("test_size", 0)
    val_rows = kwargs.get("val_size")
    mine = list(TimeSeriesCV(data, n_splits=3, **kwargs).get_validation())
    expected = list(
        TimeSeriesSplit(
            n_splits=3,
            test_size=(
                ceil(round(end * val_rows, 9))
                if isinstance(val_rows, float)
                else val_rows
            ),
            gap=kwargs.get("gap", 0),
            max_train_size=kwargs.get("max_train_size"),
        ).split(range(end))
    )[skip:]
    assert [split.index for split in mine] == list(range(skip + 1, 4))
    for split, (train, val) in zip(mine, expected, strict=True):
        assert split.train.object_id.to_list() == train.tolist()
        assert split.heldout.object_id.to_list() == val.tolist()


def test_cv_panel_skip_splits_keeps_indices():
    windows = list(
        TimeSeriesCV(
            panel_data(), n_splits=3, val_size=4, skip_splits=1
        ).get_validation()
    )
    assert [split.index for split in windows] == [2, 3]
    held = [row for split in windows for row in split.heldout.object_id.to_list()]
    assert held == list(range(15, 24))


def test_cv_collapsed_skipped_window_is_allowed():
    data = identified_data()
    dates = (
        [date(2024, 1, 1) + timedelta(days=n) for n in range(11)]
        + [date(2024, 2, 1)] * 7
        + [date(2024, 3, 1) + timedelta(days=n) for n in range(6)]
    )
    data = replace(data, time=pl.Series("date", dates))
    windows = list(
        TimeSeriesCV(data, n_splits=3, val_size=4, skip_splits=2).get_validation()
    )
    assert [split.index for split in windows] == [3]
    assert windows[0].heldout.object_id.to_list() == list(range(20, 24))
    with pytest.raises(ValueError, match="single timestamp spans more rows"):
        TimeSeriesCV(data, n_splits=3, val_size=4)


def test_cv_without_training_rows_reports_gap_error():
    with pytest.raises(ValueError, match="Not enough rows for the requested CV"):
        TimeSeriesCV(panel_data(), n_splits=2, val_size=11, gap=timedelta(days=1))


def test_tiny_float_size_keeps_one_row():
    split = next(Holdout(identified_data(), val_size=1e-12).get_validation())
    assert len(split.heldout) == 1
    assert len(Holdout(identified_data(), test_size=1e-12).get_test().heldout) == 1


@pytest.mark.parametrize(
    "dtype",
    [
        pl.Datetime("ms"),
        pl.Datetime("ns"),
        pl.Datetime("us", "Europe/Moscow"),
    ],
)
def test_zero_timedelta_gap_on_fine_timestamps(dtype):
    base = identified_data()
    times = pl.Series(
        "time",
        [datetime(2024, 1, 1) + timedelta(milliseconds=n) for n in range(24)],
    )
    if dtype.time_zone is not None:
        times = times.dt.replace_time_zone(dtype.time_zone)
    data = replace(base, time=times.cast(dtype))
    split = next(Holdout(data, val_size=5, gap=timedelta(0)).get_validation())
    assert split.train.object_id.to_list() == list(range(19))
    assert split.heldout.object_id.to_list() == list(range(19, 24))


def test_integer_test_size_snaps_to_first_row_of_timestamp():
    final = Holdout(panel_data(), test_size=4).get_test()
    assert final.heldout.object_id.to_list() == list(range(18, 24))
    assert len(final.train) == 18


@pytest.mark.parametrize("kwargs", [{"test_size": True}, {"val_size": True}])
def test_boolean_sizes_are_rejected_for_holdout(kwargs):
    with pytest.raises(ValueError, match="must be positive rows"):
        Holdout(identified_data(), **kwargs)


@pytest.mark.parametrize(
    "kwargs", [{"n_splits": True}, {"skip_splits": True}, {"max_train_size": True}]
)
def test_boolean_cv_counts_are_rejected(kwargs):
    with pytest.raises(ValueError):
        TimeSeriesCV(identified_data(), **{"n_splits": 3, **kwargs})


def test_unique_timestamps_holdout_unchanged():
    split = next(
        Holdout(identified_data(), val_size=5, gap=timedelta(0)).get_validation()
    )
    assert split.train.object_id.to_list() == list(range(19))
    assert split.heldout.object_id.to_list() == list(range(19, 24))


def test_float_size_rounding_noise_does_not_add_a_row():
    n = 100
    base = identified_data()
    data = type(base)(
        pl.DataFrame({"x": np.arange(n, dtype=float), "unused": np.zeros(n)}),
        pl.Series("target", np.arange(n, dtype=float)),
        pl.Series("date", [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]),
        pl.Series("weight", np.ones(n)),
        object_id=pl.Series("id", np.arange(n)),
        auxiliary_features=pl.Series("group", ["a", "b"] * (n // 2)).to_frame(),
        numerical_features=("x", "unused"),
    )
    assert len(Holdout(data, test_size=0.07).get_test().heldout) == 7


@pytest.mark.parametrize(
    "factory", [Holdout, lambda d, **k: TimeSeriesCV(d, n_splits=2, **k)]
)
def test_boolean_val_size_is_rejected(factory):
    with pytest.raises(ValueError, match="val_size must be positive rows"):
        factory(identified_data(), val_size=True)


def test_cv_single_timestamp_wider_than_window_is_rejected():
    with pytest.raises(ValueError, match="single timestamp spans more rows"):
        TimeSeriesCV(panel_data(), n_splits=4, val_size=1)
