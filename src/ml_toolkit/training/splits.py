"""Data-bound splitting strategies for validation and final fitting."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from math import ceil
from typing import Generic, TypeVar

import polars as pl
from sklearn.model_selection import TimeSeriesSplit

from ..data import Pool, validate_auxiliary_pair, validate_training_data

DataT = TypeVar("DataT", bound=Pool, covariant=True)


@dataclass(frozen=True)
class Split(Generic[DataT]):
    """Hold training data and an optional held-out sample.

    Parameters
    ----------
    train : Pool
        Nonempty training observations with ascending timestamps.
    heldout : Pool or None, default None
        Validation or test observations, with matching feature schema and
        metadata. None is supported for final fitting without a test sample.
    index : int or None, default None
        Positive unique fold number required for validation. Final fitting
        ignores this field. Built-in CV preserves numbers before skipping.

    Notes
    -----
    Construction validates each pool's timestamps, targets and weights and the
    pair's feature compatibility. Pools retain their concrete container type.
    The splitter is responsible for sample membership and preventing leakage.
    """

    train: DataT
    heldout: DataT | None = None
    index: int | None = None

    def __post_init__(self) -> None:
        validate_training_data(self.train)
        if self.heldout is not None:
            validate_training_data(self.heldout)
            validate_auxiliary_pair(self.train, self.heldout)
            if (
                self.train.X.schema != self.heldout.X.schema
                or self.train.cat_features != self.heldout.cat_features
                or self.train.numerical_features != self.heldout.numerical_features
            ):
                raise ValueError("Split pools must have matching features and metadata")


class Splitter(ABC, Generic[DataT]):
    """Provide validation splits and final-fit data from a prepared pool.

    Parameters
    ----------
    data : Pool
        Project-prepared observations, retained without copying. Construction
        validates chronological order, finite targets and nonnegative weights.

    Notes
    -----
    Implement both methods. Each get_validation call must return a fresh
    iterator with at least one numbered train/validation pair. Iteration order
    determines execution order and which split supplies final-fit settings.
    get_test owns all non-test rows and the optional test sample. get_final
    supplies the full eligible pool for optional training including test.
    """

    def __init__(self, data: DataT):
        validate_training_data(data)
        self.data = data

    @abstractmethod
    def get_validation(self) -> Iterator[Split[DataT]]:
        """Return a fresh iterator of numbered train/validation pairs."""
        ...

    @abstractmethod
    def get_test(self) -> Split[DataT]:
        """Return all non-test data and an optional held-out test sample."""
        ...

    def get_final(self) -> DataT:
        """Return all eligible data, including test, for an optional fresh fit."""
        return self.data

    def parameters(self) -> dict[str, object]:
        """Return explicit JSON-compatible settings for experiment logging."""
        return {}


def _validate_size(size: int | float, name: str) -> None:
    if size <= 0 or (isinstance(size, float) and not 0 < size < 1):
        raise ValueError(f"{name} must be positive rows or a fraction between 0 and 1")


def _normalize_test_size(
    size: int | float | str | date | None,
) -> int | float | date | None:
    if isinstance(size, str):
        try:
            parsed = date.fromisoformat(size)
        except ValueError as error:
            raise ValueError("test_size date must use YYYY-MM-DD format") from error
        if parsed.isoformat() != size:
            raise ValueError("test_size date must use YYYY-MM-DD format")
        return parsed
    if isinstance(size, datetime):
        raise ValueError("test_size must be a calendar date without a time")
    if isinstance(size, (int, float)):
        _validate_size(size, "test_size")
    return size


def _rows(size: int | float, total: int) -> int:
    return ceil(total * size) if isinstance(size, float) else size


def _validate_gap(gap: int | timedelta) -> None:
    if isinstance(gap, timedelta):
        if gap < timedelta(0):
            raise ValueError("gap must be nonnegative")
    elif type(gap) is not int or gap < 0:
        raise ValueError("gap must be nonnegative integer rows or timedelta")


def _gap_parameter(gap: int | timedelta) -> int | dict[str, object]:
    if isinstance(gap, int):
        return gap
    return {"type": "timedelta", "microseconds": gap // timedelta(microseconds=1)}


def _training_end(data: Pool, validation_start: int, gap: int | timedelta) -> int:
    if isinstance(gap, int):
        return validation_start - gap
    # Integer timestamps measure elapsed time even across daylight-saving changes.
    # Convert Date to midnight, preserving sub-day gaps rather than rounding them.
    time = data.time
    if time.dtype == pl.Date:
        time = time.cast(pl.Datetime("us"))
    unit = time.dtype.time_unit if isinstance(time.dtype, pl.Datetime) else "us"
    scale = {"ms": 1_000_000, "us": 1000, "ns": 1}[unit]
    duration_ns = (gap // timedelta(microseconds=1)) * 1000
    # Python integers avoid overflowing timestamp arithmetic for large gaps.
    start_ns = int(time.slice(validation_start, 1).cast(pl.Int64)[0]) * scale
    cutoff = start_ns - duration_ns
    timestamps = time.head(validation_start).cast(pl.Int64)
    left, right = 0, validation_start
    while left < right:
        middle = (left + right) // 2
        if int(timestamps[middle]) * scale <= cutoff:
            left = middle + 1
        else:
            right = middle
    return left


def _train_end(data: Pool, test_size: int | float | date | None) -> int:
    if test_size is None:
        return len(data)
    end = (
        int((data.time.dt.date() < test_size).sum())
        if isinstance(test_size, date)
        else len(data) - _rows(test_size, len(data))
    )
    if end <= 0 or end >= len(data):
        raise ValueError("test_size must leave nonempty training and test data")
    return end


def _final(data: DataT, end: int) -> Split[DataT]:
    if end == len(data):
        return Split(data)
    return Split(data.take(range(end)), data.take(range(end, len(data))))


class Holdout(Splitter[DataT]):
    """Reserve trailing validation and optional test samples.

    Parameters
    ----------
    data : Pool
        Chronologically ordered observations, including the optional test tail.
    val_size : int or float, default 0.2
        Validation row count or fraction of non-test rows, rounded upward.
    test_size : int, float, str, datetime.date, or None, default None
        Test row count, fraction of the original pool rounded upward, or an
        inclusive calendar start date in the data's timezone. Date strings use
        YYYY-MM-DD; datetimes are rejected. None disables test evaluation.
    gap : int or datetime.timedelta, default 0
        Nonnegative row gap or minimum elapsed time before validation. A row
        exactly validation_start - gap is retained for a timedelta gap.

    Notes
    -----
    get_validation yields one split with index 1. get_test includes every
    non-test row, including validation and gap rows.
    """

    def __init__(
        self,
        data: DataT,
        *,
        val_size: int | float = 0.2,
        test_size: int | float | str | date | None = None,
        gap: int | timedelta = 0,
    ):
        super().__init__(data)
        _validate_size(val_size, "val_size")
        _validate_gap(gap)
        self.val_size = val_size
        self.test_size = _normalize_test_size(test_size)
        self.gap = gap
        self._end = _train_end(data, self.test_size)
        self._val_start = self._end - _rows(val_size, self._end)
        if self._val_start <= 0:
            raise ValueError("Not enough rows for train, gap and validation")
        self._train_end = _training_end(data, self._val_start, gap)
        if self._train_end <= 0:
            raise ValueError("Not enough rows for train, gap and validation")

    def get_validation(self) -> Iterator[Split[DataT]]:
        """Yield the trailing validation window and its preceding training rows."""
        yield Split(
            self.data.take(range(self._train_end)),
            self.data.take(range(self._val_start, self._end)),
            index=1,
        )

    def get_test(self) -> Split[DataT]:
        """Return all non-test rows and the optional test tail."""
        return _final(self.data, self._end)

    def parameters(self) -> dict[str, object]:
        """Return validation size, normalized test size and gap."""
        return {
            "val_size": self.val_size,
            "test_size": (
                self.test_size.isoformat()
                if isinstance(self.test_size, date)
                else self.test_size
            ),
            "gap": _gap_parameter(self.gap),
        }


class TimeSeriesCV(Splitter[DataT]):
    """Yield expanding or bounded chronological validation windows.

    Parameters
    ----------
    data : Pool
        Chronologically ordered observations, including the optional test tail.
    n_splits : int
        Number of generated windows, at least two.
    val_size : int, float, or None, default None
        Validation row count or fraction of all non-test rows, rounded upward.
        None uses sklearn TimeSeriesSplit's equal-block default.
    test_size : int, float, str, datetime.date, or None, default None
        Test row count, fraction of the original pool, or inclusive calendar
        start date, as in Holdout. None disables test evaluation.
    gap : int or datetime.timedelta, default 0
        Nonnegative row gap or minimum elapsed time before each validation
        window. Apply max_train_size after excluding the gap.
    max_train_size : int or None, default None
        Positive maximum training rows per validation window. None expands
        training to include all preceding rows. Final fitting is unrestricted.
    skip_splits : int, default 0
        Initial windows to omit; must be nonnegative and smaller than n_splits.
        Remaining windows retain their original one-based indices.

    Notes
    -----
    get_test includes every non-test row, including validation and gap rows.
    Each iterator creates pools only for the window currently requested.
    """

    def __init__(
        self,
        data: DataT,
        *,
        n_splits: int,
        val_size: int | float | None = None,
        test_size: int | float | str | date | None = None,
        gap: int | timedelta = 0,
        max_train_size: int | None = None,
        skip_splits: int = 0,
    ):
        super().__init__(data)
        if val_size is not None:
            _validate_size(val_size, "val_size")
        _validate_gap(gap)
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if skip_splits < 0 or skip_splits >= n_splits:
            raise ValueError(
                "skip_splits must be nonnegative and smaller than n_splits"
            )
        if max_train_size is not None and max_train_size < 1:
            raise ValueError("max_train_size must be positive")
        self.n_splits = n_splits
        self.val_size = val_size
        self.test_size = _normalize_test_size(test_size)
        self.gap = gap
        self.max_train_size = max_train_size
        self.skip_splits = skip_splits
        self._end = _train_end(data, self.test_size)
        self._val_rows = None if val_size is None else _rows(val_size, self._end)
        rows = self._end // (n_splits + 1) if self._val_rows is None else self._val_rows
        row_gap = gap if isinstance(gap, int) else 0
        if n_splits + 1 > self._end or self._end - row_gap - rows * n_splits <= 0:
            raise ValueError("Not enough rows for the requested CV windows and gap")
        if isinstance(gap, timedelta):
            first_start = self._end - rows * (n_splits - skip_splits)
            if _training_end(data, first_start, gap) <= 0:
                raise ValueError("Not enough rows for the requested CV windows and gap")

    def get_validation(self) -> Iterator[Split[DataT]]:
        """Yield validation windows in order, retaining original fold numbers."""
        splitter = TimeSeriesSplit(
            n_splits=self.n_splits,
            test_size=self._val_rows,
            gap=self.gap if isinstance(self.gap, int) else 0,
            max_train_size=self.max_train_size if isinstance(self.gap, int) else None,
        )
        for index, (train, val) in enumerate(splitter.split(range(self._end)), start=1):
            if index > self.skip_splits:
                if isinstance(self.gap, timedelta):
                    end = _training_end(self.data, int(val[0]), self.gap)
                    start = (
                        0
                        if self.max_train_size is None
                        else max(0, end - self.max_train_size)
                    )
                    train = range(start, end)
                yield Split(self.data.take(train), self.data.take(val), index=index)

    def get_test(self) -> Split[DataT]:
        """Return all non-test rows and the optional test tail."""
        return _final(self.data, self._end)

    def parameters(self) -> dict[str, object]:
        """Return explicit window and test settings, without pool data."""
        return {
            "n_splits": self.n_splits,
            "val_size": self.val_size,
            "test_size": (
                self.test_size.isoformat()
                if isinstance(self.test_size, date)
                else self.test_size
            ),
            "gap": _gap_parameter(self.gap),
            "max_train_size": self.max_train_size,
            "skip_splits": self.skip_splits,
        }
