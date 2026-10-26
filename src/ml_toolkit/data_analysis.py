"""Composable data analysis reports that do not require model predictions.

Reports receive aligned rows and train/val/test labels. Instances are reused
across fits and must not retain data or mutate their inputs.
"""

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import polars as pl

from ._time import PERIOD_INTERVALS
from .config import FigureConfig
from .data import Pool
from .feature_levels import FeatureLevelConfig
from .feature_levels.config import validate_features
from .plots import monitoring
from .plots.utils import feature_filename
from .types import TimePeriod


@dataclass(frozen=True, kw_only=True)
class DataAnalyzer:
    """Named data metrics and artifacts, independent of model predictions.

    Implement either hook or both. During training, finite metrics are logged
    under ``<name>/<key>`` and only returned files are uploaded under ``<name>/``.
    These metrics do not become CV averages or optimization objectives.
    Names must be unique across analyzers and prediction evaluators.
    """

    name: str

    def __post_init__(self) -> None:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.name) is None:
            raise ValueError(
                "Analyzer name must start with an ASCII letter or digit and contain only ASCII letters, digits, underscores, hyphens or dots"
            )

    def evaluate(self, *, data: Pool, split: pl.Series) -> dict[str, float]:
        """Return finite scalar measurements, or an empty dictionary.

        Parameters
        ----------
        data : Pool
            Combined train and optional held-out rows. Do not mutate its fields.
        split : polars.Series
            Aligned non-null String labels: train plus at most one of val/test.

        Returns
        -------
        dict of str to float
            Additional values logged as ``<name>/<key>``. Include sample names
            in keys where relevant.
        """
        return {}

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
        """Write data reports and return only files to upload.

        Parameters
        ----------
        data : Pool
            Combined train and optional held-out rows; inputs must not be mutated.
        split : polars.Series
            Aligned non-null String labels: train plus at most one of val/test.
        artifact_dir : pathlib.Path
            Temporary directory owned by the caller, removed after uploading.
        figure : FigureConfig
            Shared figure dimensions, resolution and format.
        features : tuple of FeatureLevelConfig or None
            Shared feature-level rules; None infers rules and () disables levels.
            Use only if relevant to this analysis.
        n_bins : int
            Shared numerical feature-level bin limit.

        Returns
        -------
        list of pathlib.Path
            Files relative to artifact_dir, or absolute paths inside it.
            Only these files are uploaded under the analyzer name.
        """
        return []

    def parameters(self) -> dict[str, object]:
        """Return explicit JSON-compatible settings, never data or fitted state."""
        return {}


@dataclass(frozen=True, kw_only=True)
class NumericalFeatureDistributions(DataAnalyzer):
    """Compare unweighted numerical time distributions on original features.

    Parameters
    ----------
    name : str, default 'numerical_feature_distributions'
        Unique report name and artifact namespace.
    features : tuple of str or None, default None
        Numeric feature names to display. None uses numeric columns in Pool.X;
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
        names = self.features
        if names is None:
            names = tuple(
                name for name, dtype in train.X.schema.items() if dtype.is_numeric()
            )
        time_column = "__ml_toolkit_plot_time__"
        while time_column in train.X.columns:
            time_column += "_"
        reference = train.X.with_columns(train.time.alias(time_column))
        heldout = current.X.with_columns(current.time.alias(time_column))
        for index, name in enumerate(names):
            if name not in train.X.columns or not train.X.schema[name].is_numeric():
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


def validate_split(data: Pool, split: pl.Series) -> None:
    if (
        len(split) != len(data)
        or split.dtype != pl.String
        or split.null_count()
        or not split.is_in(("train", "val", "test")).all()
    ):
        raise ValueError(
            "split must contain aligned non-null String train/val/test labels"
        )
    if {"val", "test"} <= set(split.unique()):
        raise ValueError(
            "Reporting accepts only one held-out split: val or test, not both"
        )


def data_comparison_pair(data: Pool, split: pl.Series) -> tuple[str, Pool, Pool] | None:
    """Select train and a single held-out sample without requiring predictions."""
    validate_split(data, split)
    train = np.flatnonzero((split == "train").to_numpy())
    current = np.flatnonzero((split != "train").to_numpy())
    if not len(train) or not len(current):
        return None
    return str(split[int(current[0])]), data.take(train), data.take(current)


__all__ = ["DataAnalyzer", "NumericalFeatureDistributions"]
