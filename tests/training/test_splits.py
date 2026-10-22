"""Temporal split membership, repeatability, metadata and boundary validation."""

from dataclasses import replace
from datetime import date, datetime, timedelta

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
            for name in ("y", "time", "weight", "business_type", "object_id"):
                assert getattr(part, name).equals(getattr(data, name).gather(indices))
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
