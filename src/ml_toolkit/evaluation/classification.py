"""Binary-classification probability and threshold reports."""

from dataclasses import dataclass
from pathlib import Path
import polars as pl
from .base import PredictionEvaluator
from ..config import FigureConfig
from ..data import Pool
from ..feature_levels import FeatureLevelConfig
from ..plots import classification
from ..types import FloatArray

from ._common import write_heldout_plot


@dataclass(frozen=True, kw_only=True)
class PrecisionRecall(PredictionEvaluator):
    """Plot weighted precision against recall on held-out probabilities.

    Parameters
    ----------
    name : str, default 'precision_recall'
        Unique report name and artifact namespace.
    max_curve_points : int, default 1000
        Maximum displayed threshold points; does not change metric calculations.

    Notes
    -----
    Uses the single validation or test sample, ignoring train rows. Targets are
    confirmed binary labels and predictions are probabilities in [0, 1].
    Saves ``<split>/precision_recall.<format>``. Row weights affect both curves.
    """

    name: str = "precision_recall"
    max_curve_points: int = 1000

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.max_curve_points < 1:
            raise ValueError("max_curve_points must be positive")

    def parameters(self) -> dict[str, object]:
        return {"max_curve_points": self.max_curve_points}

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
            "precision_recall",
            classification.precision_recall,
            max_curve_points=self.max_curve_points,
        )


@dataclass(frozen=True, kw_only=True)
class F1Threshold(PredictionEvaluator):
    """Plot weighted F1 against held-out probability thresholds.

    Parameters
    ----------
    name : str, default 'f1_threshold'
        Unique report name and artifact namespace.
    max_curve_points : int, default 1000
        Maximum displayed threshold points.

    Notes
    -----
    Uses confirmed binary validation or test labels and probabilities in [0, 1].
    Saves ``<split>/f1_threshold.<format>`` without changing the model threshold.
    """

    name: str = "f1_threshold"
    max_curve_points: int = 1000

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.max_curve_points < 1:
            raise ValueError("max_curve_points must be positive")

    def parameters(self) -> dict[str, object]:
        return {"max_curve_points": self.max_curve_points}

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
            "f1_threshold",
            classification.f1_threshold,
            max_curve_points=self.max_curve_points,
        )


@dataclass(frozen=True, kw_only=True)
class Calibration(PredictionEvaluator):
    """Compare held-out probabilities with observed positive rates.

    Parameters
    ----------
    name : str, default 'calibration'
        Unique report name and artifact namespace.
    n_bins : int, default 10
        Number of probability quantiles for the calibration curve. Independent
        of shared feature-level binning.
    histogram_bins : int, default 30
        Number of bins in the accompanying probability histogram.

    Notes
    -----
    Uses confirmed binary labels and probabilities in [0, 1]. Means and the
    histogram use weights; quantile boundaries use positive-weight rows without
    weighting. Saves ``<split>/calibration.<format>`` for validation or test.
    """

    name: str = "calibration"
    n_bins: int = 10
    histogram_bins: int = 30

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_bins < 1:
            raise ValueError("n_bins must be positive")
        if self.histogram_bins < 1:
            raise ValueError("histogram_bins must be positive")

    def parameters(self) -> dict[str, object]:
        return {"n_bins": self.n_bins, "histogram_bins": self.histogram_bins}

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
            "calibration",
            classification.calibration,
            n_bins=self.n_bins,
            histogram_bins=self.histogram_bins,
        )
