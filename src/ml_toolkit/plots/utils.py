"""Input checks and figure handling shared by the plot families."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..config import FigureConfig

logger = logging.getLogger(__name__)


class PlotError(ValueError):
    """Invalid data for a diagnostic plot."""


def numeric_vector(values, name: str, *, probability: bool = False) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise PlotError(f"{name} must be a nonempty, finite one-dimensional array")
    if probability and ((result < 0).any() or (result > 1).any()):
        raise PlotError(f"{name} must contain probabilities in [0, 1]")
    return result


def same_length(**vectors) -> None:
    if len({len(value) for value in vectors.values()}) != 1:
        raise PlotError(f"{', '.join(vectors)} must have the same length")


def weights(values, length: int) -> np.ndarray:
    result = np.ones(length) if values is None else numeric_vector(values, "weight")
    if len(result) != length or (result < 0).any() or not (result > 0).any():
        raise PlotError("weight must match the data and have positive total exposure")
    return result


def save_figure(fig, path: str | Path, *, config: FigureConfig | None = None) -> Path:
    """Create parent directories, save a plot, and close its figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to export; always closed, including when saving fails.
    path : str or pathlib.Path
        Output path. An explicit config replaces the suffix with its format.
    config : FigureConfig or None, default None
        Dimensions, DPI, and format. None preserves native dimensions and the
        path's format, using Matplotlib's default resolution.

    Returns
    -------
    pathlib.Path
        Actual output path, including any configured suffix replacement.
    """
    path = Path(path)
    if config is not None:
        path = path.with_suffix(f".{config.format}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if config is not None and config.figsize is not None:
            fig.set_size_inches(*config.figsize)
            fig.tight_layout()
        settings = (
            {} if config is None else {"dpi": config.dpi, "format": config.format}
        )
        fig.savefig(path, bbox_inches="tight", **settings)
    finally:
        plt.close(fig)
    logger.debug("Saved figure to %s", path)
    return path
