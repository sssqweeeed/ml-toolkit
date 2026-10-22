"""Shared input and feature helpers for prediction and data reports."""

import re

import numpy as np
import polars as pl

from .data import Pool
from .feature_levels import FeatureLevelConfig


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


def feature_definitions(
    data: Pool, features: tuple[FeatureLevelConfig, ...] | None
) -> tuple[FeatureLevelConfig, ...]:
    if features is not None:
        return features
    return tuple(
        FeatureLevelConfig(
            name,
            (
                "categorical"
                if name in data.cat_features or not dtype.is_numeric()
                else "numerical"
            ),
        )
        for name, dtype in data.X.schema.items()
    )


def feature_filename(index: int, feature: str, format: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", feature).strip("._")[:60] or "feature"
    return f"{index:03d}_{safe}.{format}"


def validate_features(features: tuple[str, ...] | None) -> None:
    if features is not None and (
        any(not name.strip() for name in features)
        or len(set(features)) != len(features)
    ):
        raise ValueError("features must contain unique nonempty names")
