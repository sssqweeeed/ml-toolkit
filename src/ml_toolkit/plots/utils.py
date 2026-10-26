"""Input checks and figure handling shared by the plot families."""

from collections.abc import Sized
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from .._console import Logger
from ..config import FigureConfig
from ..types import FloatArray

logger = Logger(__name__)


class PlotError(ValueError):
    """Invalid data for a diagnostic plot."""


def numeric_vector(
    values: ArrayLike, name: str, *, probability: bool = False
) -> FloatArray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise PlotError(f"{name} must be a nonempty, finite one-dimensional array")
    if probability and ((result < 0).any() or (result > 1).any()):
        raise PlotError(f"{name} must contain probabilities in [0, 1]")
    return result


def same_length(**vectors: Sized) -> None:
    if len({len(value) for value in vectors.values()}) != 1:
        raise PlotError(f"{', '.join(vectors)} must have the same length")


def weights(values: ArrayLike | None, length: int) -> FloatArray:
    result = np.ones(length) if values is None else numeric_vector(values, "weight")
    if len(result) != length or (result < 0).any() or not (result > 0).any():
        raise PlotError("weight must match the data and have positive total exposure")
    return result


def save_figure(
    fig: Figure, path: str | Path, *, config: FigureConfig | None = None
) -> Path:
    """Save and close a figure, returning the path after any format replacement.

    Creates parent directories. An explicit config applies its size, DPI, and
    format; None preserves native dimensions and uses the path's format.
    The figure is closed even when saving fails.
    """
    path = Path(path)
    if config is not None:
        path = path.with_suffix(f".{config.format}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if config is not None and config.figsize is not None:
            fig.set_size_inches(*config.figsize)
            fig.tight_layout()
        if config is None:
            fig.savefig(path, bbox_inches="tight")
        else:
            fig.savefig(path, bbox_inches="tight", dpi=config.dpi, format=config.format)
    finally:
        plt.close(fig)
    logger.debug("Saved figure to %s", path)
    return path


def feature_filename(index: int, feature: str, format: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", feature).strip("._")[:60] or "feature"
    return f"{index:03d}_{safe}.{format}"
