"""Outcome capture and lift reports on held-out predictions."""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import polars as pl
from .base import PredictionEvaluator
from ..config import FigureConfig
from ..data import Pool
from ..feature_levels import FeatureLevelConfig
from ..metrics.ranking import ranking_metrics
from ..plots import ranking
from ..types import FloatArray

from ._common import write_heldout_plot


def _gains(
    observed: FloatArray,
    prediction: FloatArray,
    *,
    weight: FloatArray,
    save_fig: Path,
    figure: FigureConfig,
) -> None:
    if not np.all(observed >= 0) or np.sum(observed * weight) <= 0:
        raise ValueError(
            "Gains/lift require nonnegative targets with positive weighted total"
        )
    ranking.gains_curve(
        ranking_metrics(observed, prediction, weight), save_fig=save_fig, figure=figure
    )


def _lift(
    observed: FloatArray,
    prediction: FloatArray,
    *,
    weight: FloatArray,
    save_fig: Path,
    figure: FigureConfig,
) -> None:
    if not np.all(observed >= 0) or np.sum(observed * weight) <= 0:
        raise ValueError(
            "Gains/lift require nonnegative targets with positive weighted total"
        )
    ranking.lift_curve(
        ranking_metrics(observed, prediction, weight), save_fig=save_fig, figure=figure
    )


@dataclass(frozen=True, kw_only=True)
class Gains(PredictionEvaluator):
    """Plot captured observed outcome against held-out exposure capacity.

    Parameters
    ----------
    name : str, default 'gains'
        Unique report name and artifact namespace.

    Notes
    -----
    Ranks validation or test rows by descending prediction. Pool weights supply
    exposure; targets must be nonnegative with positive weighted total. The
    curve shows cumulative ``target * weight`` share, including the default 10%
    capacity point. Saves ``<split>/gains.<format>``.
    """

    name: str = "gains"

    def write_artifacts(
        self,
        *,
        data: Pool,
        prediction: FloatArray,
        split: pl.Series,
        artifact_dir: Path,
        figure: FigureConfig,
        features: tuple[FeatureLevelConfig, ...] | None,
        n_bins: int,
    ) -> list[Path]:
        return write_heldout_plot(
            data, prediction, split, artifact_dir, figure, "gains", _gains
        )


@dataclass(frozen=True, kw_only=True)
class Lift(PredictionEvaluator):
    """Plot held-out outcome capture relative to uniform selection.

    Parameters
    ----------
    name : str, default 'lift'
        Unique report name and artifact namespace.

    Notes
    -----
    Lift divides captured weighted outcome share by exposure share after
    ranking by prediction. Uses validation or test rows and the default 10%
    capacity point. Targets must be nonnegative with positive weighted total.
    Saves ``<split>/lift.<format>``.
    """

    name: str = "lift"

    def write_artifacts(
        self,
        *,
        data: Pool,
        prediction: FloatArray,
        split: pl.Series,
        artifact_dir: Path,
        figure: FigureConfig,
        features: tuple[FeatureLevelConfig, ...] | None,
        n_bins: int,
    ) -> list[Path]:
        return write_heldout_plot(
            data, prediction, split, artifact_dir, figure, "lift", _lift
        )
