"""Aligned prediction validation and shared split selection."""

from collections.abc import Callable
from pathlib import Path
import numpy as np
import polars as pl
from .._rows import row_indexer
from ..config import FigureConfig
from ..data import Pool
from ..data_analysis._common import validate_split
from ..metrics._evaluation import validate_prediction
from ..types import FloatArray


def validate_inputs(data: Pool, prediction: FloatArray, split: pl.Series) -> None:
    validate_prediction(data, prediction)
    validate_split(data, split)


def _heldout(
    data: Pool, prediction: FloatArray, split: pl.Series
) -> tuple[str, Pool, FloatArray] | None:
    validate_inputs(data, prediction, split)
    index = np.flatnonzero((split != "train").to_numpy())
    if not len(index):
        return None
    part = str(split[int(index[0])])
    return part, data.take(index), prediction[row_indexer(index, len(data))]


def comparison_pair(
    data: Pool, prediction: FloatArray, split: pl.Series
) -> tuple[str, Pool, FloatArray, Pool, FloatArray] | None:
    """Select the single train/held-out pair, or skip an incomplete pair."""
    current = _heldout(data, prediction, split)
    index = np.flatnonzero((split == "train").to_numpy())
    if current is None or not len(index):
        return None
    part, heldout, current_prediction = current
    return (
        part,
        data.take(index),
        prediction[row_indexer(index, len(data))],
        heldout,
        current_prediction,
    )


def write_heldout_plot(
    data: Pool,
    prediction: FloatArray,
    split: pl.Series,
    root: Path,
    figure: FigureConfig,
    filename: str,
    draw: Callable[..., object],
    **kwargs: object,
) -> list[Path]:
    pair = _heldout(data, prediction, split)
    if pair is None:
        return []
    part, current, predicted = pair
    path = root / part / f"{filename}.{figure.format}"
    draw(
        current.y.to_numpy(),
        predicted,
        save_fig=path,
        weight=current.weight.to_numpy(),
        figure=figure,
        **kwargs,
    )
    return [path]
