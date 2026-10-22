"""Data-bound splitting strategies for validation and final fitting."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime
from math import ceil
from typing import Generic, TypeVar

from sklearn.model_selection import TimeSeriesSplit

from ..data import Pool, validate_training_data

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
    get_final owns the final training rows and optional test sample.
    """

    def __init__(self, data: DataT):
        validate_training_data(data)
        self.data = data

    @abstractmethod
    def get_validation(self) -> Iterator[Split[DataT]]:
        """Return a fresh iterator of numbered train/validation pairs."""
        ...

    @abstractmethod
    def get_final(self) -> Split[DataT]:
        """Return final training data and optional test data."""
        ...

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
    gap : int, default 0
        Nonnegative number of rows excluded immediately before validation.

    Notes
    -----
    get_validation yields one split with index 1. get_final includes every
    non-test row, including validation and gap rows.
    """

    def __init__(
        self,
        data: DataT,
        *,
        val_size: int | float = 0.2,
        test_size: int | float | str | date | None = None,
        gap: int = 0,
    ):
        super().__init__(data)
        _validate_size(val_size, "val_size")
        if gap < 0:
            raise ValueError("gap must be nonnegative")
        self.val_size = val_size
        self.test_size = _normalize_test_size(test_size)
        self.gap = gap
        self._end = _train_end(data, self.test_size)
        self._val_start = self._end - _rows(val_size, self._end)
        if self._val_start - gap <= 0:
            raise ValueError("Not enough rows for train, gap and validation")

    def get_validation(self) -> Iterator[Split[DataT]]:
        """Yield the trailing validation window and its preceding training rows."""
        yield Split(
            self.data.take(range(self._val_start - self.gap)),
            self.data.take(range(self._val_start, self._end)),
            index=1,
        )

    def get_final(self) -> Split[DataT]:
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
            "gap": self.gap,
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
    gap : int, default 0
        Nonnegative row gap immediately before each validation window.
    max_train_size : int or None, default None
        Positive maximum training rows per validation window. None expands
        training to include all preceding rows. Final fitting is unrestricted.
    skip_splits : int, default 0
        Initial windows to omit; must be nonnegative and smaller than n_splits.
        Remaining windows retain their original one-based indices.

    Notes
    -----
    get_final includes every non-test row, including validation and gap rows.
    Each iterator creates pools only for the window currently requested.
    """

    def __init__(
        self,
        data: DataT,
        *,
        n_splits: int,
        val_size: int | float | None = None,
        test_size: int | float | str | date | None = None,
        gap: int = 0,
        max_train_size: int | None = None,
        skip_splits: int = 0,
    ):
        super().__init__(data)
        if val_size is not None:
            _validate_size(val_size, "val_size")
        if gap < 0:
            raise ValueError("gap must be nonnegative")
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
        if n_splits + 1 > self._end or self._end - gap - rows * n_splits <= 0:
            raise ValueError("Not enough rows for the requested CV windows and gap")

    def get_validation(self) -> Iterator[Split[DataT]]:
        """Yield validation windows in order, retaining original fold numbers."""
        splitter = TimeSeriesSplit(
            n_splits=self.n_splits,
            test_size=self._val_rows,
            gap=self.gap,
            max_train_size=self.max_train_size,
        )
        for index, (train, val) in enumerate(splitter.split(range(self._end)), start=1):
            if index > self.skip_splits:
                yield Split(self.data.take(train), self.data.take(val), index=index)

    def get_final(self) -> Split[DataT]:
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
            "gap": self.gap,
            "max_train_size": self.max_train_size,
            "skip_splits": self.skip_splits,
        }
