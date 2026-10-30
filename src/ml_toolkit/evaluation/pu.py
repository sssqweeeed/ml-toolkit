"""Reports for known-positive and unlabeled predictions."""

from dataclasses import dataclass
from pathlib import Path
import polars as pl
from .base import PredictionEvaluator
from ..config import FigureConfig
from ..data import Pool
from ..feature_levels import FeatureLevelConfig
from ..plots import pu
from ..types import FloatArray

from ._common import write_heldout_plot


def _pu_distribution(
    observed: FloatArray,
    prediction: FloatArray,
    *,
    weight: FloatArray,
    save_fig: Path,
    figure: FigureConfig,
    histogram_bins: int,
) -> None:
    pu.score_distribution(
        observed,
        prediction,
        weights=weight,
        save_fig=save_fig,
        figure=figure,
        histogram_bins=histogram_bins,
    )


def _pu_capture(
    observed: FloatArray,
    prediction: FloatArray,
    *,
    weight: FloatArray,
    save_fig: Path,
    figure: FigureConfig,
) -> None:
    pu.positive_capture(
        observed, prediction, weights=weight, save_fig=save_fig, figure=figure
    )


@dataclass(frozen=True, kw_only=True)
class PUScoreDistribution(PredictionEvaluator):
    """Compare score distributions for known positives and unlabeled rows.

    Parameters
    ----------
    name : str, default 'pu_score_distribution'
        Unique report name and artifact namespace.
    histogram_bins : int, default 30
        Number of score bins.

    Notes
    -----
    Labels 1 and 0 mean known positive and unlabeled. Each group's weighted
    histogram is normalized separately; both need positive total weight.
    Uses validation or test rows and saves
    ``<split>/pu_score_distribution.<format>``.
    """

    name: str = "pu_score_distribution"
    histogram_bins: int = 30

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.histogram_bins < 1:
            raise ValueError("histogram_bins must be positive")

    def parameters(self) -> dict[str, object]:
        return {"histogram_bins": self.histogram_bins}

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
            data,
            prediction,
            split,
            artifact_dir,
            figure,
            "pu_score_distribution",
            _pu_distribution,
            histogram_bins=self.histogram_bins,
        )


@dataclass(frozen=True, kw_only=True)
class PUPositiveCapture(PredictionEvaluator):
    """Plot known-positive capture against unlabeled review capacity.

    Parameters
    ----------
    name : str, default 'pu_positive_capture'
        Unique report name and artifact namespace.

    Notes
    -----
    Labels 1 and 0 mean known positive and unlabeled. Both axes use weighted
    percentages; tied scores enter together. Capture is not recall for all
    positives without assumptions about labeling. Uses validation or test rows
    and saves ``<split>/pu_positive_capture.<format>``.
    """

    name: str = "pu_positive_capture"

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
            data,
            prediction,
            split,
            artifact_dir,
            figure,
            "pu_positive_capture",
            _pu_capture,
        )
