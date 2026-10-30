"""Numerical feature distributions over recorded calendar periods."""

from dataclasses import dataclass
from pathlib import Path
import polars as pl
from .._time import PERIOD_INTERVALS
from ..config import FigureConfig
from ..data import Pool
from ..feature_levels import FeatureLevelConfig
from ..feature_levels.config import validate_features
from ..plots import monitoring
from ..plots.utils import feature_filename
from ..types import TimePeriod

from .base import DataAnalyzer
from ._common import data_comparison_pair


@dataclass(frozen=True, kw_only=True)
class NumericalFeatureDistributions(DataAnalyzer):
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

    def __post_init__(self) -> None:
        super().__post_init__()
        validate_features(self.features)
        monitoring.validate_distribution(self.quantile_bands)
        if self.period not in PERIOD_INTERVALS:
            raise ValueError("Unsupported time period")

    def parameters(self) -> dict[str, object]:
        return {
            "features": self.features,
            "period": self.period,
            "quantile_bands": self.quantile_bands,
            "show_mean": self.show_mean,
            "show_median": self.show_median,
        }

    def write_artifacts(
        self,
        *,
        data: Pool,
        split: pl.Series,
        artifact_dir: Path,
        figure: FigureConfig,
        features: tuple[FeatureLevelConfig, ...] | None,
        n_bins: int,
    ) -> list[Path]:
        paths: list[Path] = []
        pair = data_comparison_pair(data, split)
        if pair is None:
            return []
        _, train, current = pair
        reference = train.to_analysis_frame()
        heldout = current.to_analysis_frame()
        names = self.features
        if names is None:
            names = tuple(
                name for name, dtype in reference.schema.items() if dtype.is_numeric()
            )
        time_column = "__ml_toolkit_plot_time__"
        while time_column in reference.columns:
            time_column += "_"
        reference = reference.with_columns(train.time.alias(time_column))
        heldout = heldout.with_columns(current.time.alias(time_column))
        for index, name in enumerate(names):
            if name not in reference.columns or not reference.schema[name].is_numeric():
                raise ValueError(f"Feature {name!r} is missing or not numerical")
            path = artifact_dir / feature_filename(index, name, figure.format)
            monitoring.time_feature_distribution(
                reference,
                heldout,
                name,
                time_column,
                save_fig=path,
                period=self.period,
                quantile_bands=self.quantile_bands,
                show_mean=self.show_mean,
                show_median=self.show_median,
                figure=figure,
            )
            paths.append(path)
        return paths
