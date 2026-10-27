"""Exact-value row groups shared by diagnostic reports."""

from dataclasses import dataclass

import numpy as np
import polars as pl

from .data import Pool
from .feature_levels.config import validate_features
from .types import IntArray


@dataclass(frozen=True)
class RowGroup:
    key: pl.DataFrame
    indices: IntArray
    identifier: str
    label: str


def validate_grouping(by: tuple[str, ...], max_groups: int, *, required: bool) -> None:
    validate_features(by)
    if required and not by:
        raise ValueError("by must contain at least one grouping column")
    if type(max_groups) is not int or max_groups < 1:
        raise ValueError("max_groups must be a positive integer")


def row_groups(data: Pool, by: tuple[str, ...], max_groups: int) -> list[RowGroup]:
    """Partition positions by typed values; null/NaN form one missing group.

    IDs are ordinal within a report and never derived from user labels. Labels
    include that ID so different typed keys cannot collapse after formatting.
    """
    frame = data.to_analysis_frame(by)
    if any(dtype.is_nested() or dtype == pl.Object for dtype in frame.schema.values()):
        raise ValueError("Grouping columns must contain scalar values")
    frame = frame.with_columns(
        pl.col(name).fill_nan(None)
        for name, dtype in frame.schema.items()
        if dtype.is_float()
    )
    # Check cardinality before materializing a potentially huge list of groups.
    if frame.n_unique() > max_groups:
        raise ValueError(
            f"Grouping exceeds max_groups={max_groups}; use coarser groups or precomputed bands"
        )
    position = "__ml_toolkit_group_row__"
    while position in frame.columns:
        position += "_"
    parts = frame.with_row_index(position).partition_by(list(by), maintain_order=True)
    groups: list[RowGroup] = []
    for index, part in enumerate(parts):
        key = part.select(by).head(1)
        identifier = f"group_{index:04d}"
        description = ", ".join(
            f"{name}={value!r}" for name, value in key.row(0, named=True).items()
        )
        groups.append(
            RowGroup(
                key=key,
                indices=part[position].to_numpy().astype(np.int64),
                identifier=identifier,
                label=f"{identifier}: {description}",
            )
        )
    return groups
