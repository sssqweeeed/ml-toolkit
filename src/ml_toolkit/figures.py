"""Figure settings, saving and input checks shared by all figure functions."""

import re
from collections.abc import Generator, Sized
from contextlib import contextmanager
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from ._console import Logger
from .types import FloatArray

logger = Logger(__name__)


@dataclass(frozen=True, kw_only=True, slots=True)
class FigureConfig:
    """Configure figure dimensions and file output.

    Parameters
    ----------
    figsize : tuple[float, float] or None, default None
        Positive finite width and height in inches. None preserves each plot's
        native or adaptive dimensions.
    dpi : int, default 150
        Positive export resolution in dots per inch.
    format : {'png', 'svg', 'pdf'}, default 'png'
        Output format. Explicit figure settings replace the output path suffix
        to match this format. All figures are closed after saving.
    """

    figsize: tuple[float, float] | None = None
    dpi: int = 150
    format: Literal["png", "svg", "pdf"] = "png"

    def __post_init__(self) -> None:
        if self.dpi < 1:
            raise ValueError("dpi must be positive")
        if self.figsize is not None and any(
            not isfinite(value) or value <= 0 for value in self.figsize
        ):
            raise ValueError("figsize dimensions must be positive")


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


@contextmanager
def saved_figure(
    fig: Figure, path: str | Path, *, config: FigureConfig | None = None
) -> Generator[Figure, None, None]:
    """Save the figure when the block succeeds and always close it."""
    try:
        yield fig
        save_figure(fig, path, config=config)
    finally:
        plt.close(fig)


def feature_filename(index: int, feature: str, format: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", feature).strip("._")[:60] or "feature"
    return f"{index:03d}_{safe}.{format}"
