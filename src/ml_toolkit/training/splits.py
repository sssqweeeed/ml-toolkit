"""Chronological partitions returned as ready-to-use training data."""

from dataclasses import dataclass
from math import ceil
from typing import Generic, TypeVar

from sklearn.model_selection import TimeSeriesSplit

from ..config import HoldoutConfig, TimeSeriesCVConfig
from ..data import validate_training_data
from .contracts import TrainingData

DataT = TypeVar("DataT", bound=TrainingData, covariant=True)


@dataclass(frozen=True)
class TrainValSplit(Generic[DataT]):
    """Hold the aligned datasets for one chronological validation split.

    Attributes
    ----------
    index : int
        One-based original split number before skipping.
    train, val : TrainingData
        Training and validation observations, preserving their container type.

    Notes
    -----
    Construction checks aligned rows, valid dates, finite targets and positive
    total weight in each window. The original container owns its row operations.
    """

    index: int
    train: DataT
    val: DataT

    def __post_init__(self) -> None:
        validate_training_data(self.train)
        validate_training_data(self.val)


def split_holdout(data: DataT, config: HoldoutConfig) -> TrainValSplit[DataT]:
    """Reserve a trailing holdout and omit the preceding gap rows.

    Parameters
    ----------
    data : TrainingData
        Chronologically ordered input with aligned metadata.
    config : HoldoutConfig
        Row count or fraction of this input, rounded upward, and a row gap.

    Returns
    -------
    TrainValSplit
        Ready training and holdout data; the split index is one.

    Raises
    ------
    ValueError
        No training rows remain after the holdout and gap.
    """
    size = config.val_size
    val_rows = ceil(len(data) * size) if isinstance(size, float) else size
    val_start = len(data) - val_rows
    train_end = val_start - config.gap
    if train_end <= 0:
        raise ValueError("Not enough rows for train, gap and validation")
    return TrainValSplit(
        1, data.take(range(train_end)), data.take(range(val_start, len(data)))
    )


def make_validation_splits(
    data: DataT,
    config: HoldoutConfig | TimeSeriesCVConfig,
) -> list[TrainValSplit[DataT]]:
    """Build ready datasets for holdout or time-series validation.

    Parameters
    ----------
    data : TrainingData
        Training observations excluding any future test tail.
    config : HoldoutConfig or TimeSeriesCVConfig
        Chronological split sizes, gaps and optional training window.

    Returns
    -------
    list[TrainValSplit]
        Evaluated splits in chronological order, preserving original indices.
    """
    if isinstance(config, HoldoutConfig):
        return [split_holdout(data, config)]
    size = config.val_size
    val_rows = ceil(len(data) * size) if isinstance(size, float) else size
    splitter = TimeSeriesSplit(
        n_splits=config.n_splits,
        test_size=val_rows,
        gap=config.gap,
        max_train_size=config.max_train_size,
    )
    return [
        TrainValSplit(index, data.take(train), data.take(val))
        for index, (train, val) in enumerate(splitter.split(data.X), start=1)
        if index > config.skip_splits
    ]
