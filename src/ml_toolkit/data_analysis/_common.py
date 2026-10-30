"""Aligned split validation and data-only comparison selection."""

import numpy as np
import polars as pl
from ..data import Pool


def validate_split(data: Pool, split: pl.Series) -> None:
    if (
        len(split) != len(data)
        or split.dtype != pl.String
        or split.null_count()
        or not split.is_in(("train", "val", "test")).all()
    ):
        raise ValueError(
            "split must contain aligned non-null String train/val/test labels"
        )
    if {"val", "test"} <= set(split.unique()):
        raise ValueError(
            "Reporting accepts only one held-out split: val or test, not both"
        )


def data_comparison_pair(data: Pool, split: pl.Series) -> tuple[str, Pool, Pool] | None:
    """Select train and a single held-out sample without requiring predictions."""
    validate_split(data, split)
    train = np.flatnonzero((split == "train").to_numpy())
    current = np.flatnonzero((split != "train").to_numpy())
    if not len(train) or not len(current):
        return None
    return str(split[int(current[0])]), data.take(train), data.take(current)
