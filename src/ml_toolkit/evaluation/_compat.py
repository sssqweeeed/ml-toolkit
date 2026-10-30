"""Compatibility adapter for prediction-based feature-distribution reports."""

from dataclasses import dataclass
from pathlib import Path
import polars as pl
from .base import PredictionEvaluator
from ..config import FigureConfig
from ..data import Pool
from ..data_analysis import (
    NumericalFeatureDistributions as _NumericalFeatureDistributions,
)
from ..feature_levels import FeatureLevelConfig
from ..types import FloatArray, TimePeriod

from ._common import validate_inputs


@dataclass(frozen=True, kw_only=True)
class NumericalFeatureDistributions(PredictionEvaluator):
    """Compare unweighted numerical time distributions on original features.

    Parameters
    ----------
    name : str, default 'numerical_feature_distributions'
        Unique report name and artifact namespace.
    features : tuple of str or None, default None
        Numeric feature names to display. None uses model and auxiliary numeric columns;
        an empty tuple selects none. Missing or nonnumeric names raise ValueError.
    period : TimePeriod, default 'month'
        Calendar grouping in the data timezone.
    quantile_bands : tuple of pairs, default ((0.05, 0.95), (0.25, 0.75))
        Lower and upper quantiles for shaded distribution bands.
    show_mean, show_median : bool, default True
        Whether to draw each central summary.

    Notes
    -----
    Compatibility adapter for existing prediction_evaluators calls. New code
    should import this report from ml_toolkit.data_analysis and pass it through
    data_analyzers instead.

    Compares train with validation or test using unweighted original values,
    independent of predictions, Pool.weight, shared feature rules, and n_bins.
    Nulls and non-finite values are omitted. Saves one indexed feature
    filename in the report root.
    """

    name: str = "numerical_feature_distributions"
    features: tuple[str, ...] | None = None
    period: TimePeriod = "month"
    quantile_bands: tuple[tuple[float, float], ...] = ((0.05, 0.95), (0.25, 0.75))
    show_mean: bool = True
    show_median: bool = True

    def _analyzer(self) -> _NumericalFeatureDistributions:
        return _NumericalFeatureDistributions(
            name=self.name,
            features=self.features,
            period=self.period,
            quantile_bands=self.quantile_bands,
            show_mean=self.show_mean,
            show_median=self.show_median,
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        self._analyzer()

    def parameters(self) -> dict[str, object]:
        return self._analyzer().parameters()

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
        validate_inputs(data, prediction, split)
        return self._analyzer().write_artifacts(
            data=data,
            split=split,
            artifact_dir=artifact_dir,
            figure=figure,
            features=features,
            n_bins=n_bins,
        )
