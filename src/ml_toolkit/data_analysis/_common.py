"""Aligned split validation and data-only comparison selection."""

import numpy as np
import polars as pl

from ..data import Pool, validate_split_labels


def data_comparison_pair(data: Pool, split: pl.Series) -> tuple[str, Pool, Pool] | None:
    """Select train and a single held-out sample without requiring predictions."""
    validate_split_labels(data, split)
    train = np.flatnonzero((split == "train").to_numpy())
    current = np.flatnonzero((split != "train").to_numpy())
    if not len(train) or not len(current):
        return None
    return str(split[int(current[0])]), data.take(train), data.take(current)
