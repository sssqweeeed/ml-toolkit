"""Fit PSI levels once and compare any Polars feature tables."""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from .config import PSIConfig
from .distributions import (
    MONTHLY_SCHEMA,
    level_distributions,
    monthly_level_distributions,
)
from .levels import LevelTransformer
from .results import PSIResult
from .score import psi


def _resolve_time(data: pl.DataFrame, time: pl.Series | str | None) -> pl.Series | None:
    if time is None:
        return None
    if isinstance(time, str):
        time = data.get_column(time)
    if not isinstance(time, pl.Series) or len(time) != len(data):
        raise ValueError("time must be a Polars Series with one value per row")
    if time.null_count():
        raise ValueError("time must not contain null values")
    if time.dtype != pl.Date and time.dtype.base_type() != pl.Datetime:
        raise TypeError("time must have Date or Datetime dtype")
    return time


@dataclass
class PSIAnalysis:
    config: PSIConfig
    transformer: LevelTransformer = field(init=False)
    reference_levels: pl.DataFrame | None = field(init=False, default=None)
    reference_time: pl.Series | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.transformer = LevelTransformer(self.config.features, n_bins=self.config.n_bins)

    def fit(
        self,
        reference: pl.DataFrame,
        *,
        time: pl.Series | str | None = None,
    ) -> PSIAnalysis:
        """Learn feature levels from reference data and optionally record dates."""
        if not isinstance(reference, pl.DataFrame) or reference.is_empty():
            raise ValueError("reference must be a non-empty Polars DataFrame")
        reference_time = _resolve_time(reference, time)
        self.transformer.fit(reference)
        self.reference_levels = self.transformer.transform(reference)
        self.reference_time = reference_time
        return self

    def compare(
        self,
        current: pl.DataFrame,
        *,
        time: pl.Series | str | None = None,
    ) -> PSIResult:
        """Score current data with the fitted reference levels."""
        if self.reference_levels is None or self.transformer.config is None:
            raise RuntimeError("PSIAnalysis must be fitted before compare")
        if not isinstance(current, pl.DataFrame) or current.is_empty():
            raise ValueError("current must be a non-empty Polars DataFrame")
        current_time = _resolve_time(current, time)
        if (self.reference_time is None) != (current_time is None):
            raise ValueError("Both samples must provide time for monthly distributions")

        current_levels = self.transformer.transform(current)
        level_config = self.transformer.config
        feature_scores = pl.DataFrame(
            {
                "feature": [feature.name for feature in level_config.features],
                "psi": [
                    psi(
                        self.reference_levels.get_column(feature.name),
                        current_levels.get_column(feature.name),
                        epsilon=self.config.epsilon,
                    )
                    for feature in level_config.features
                ],
            },
            schema={"feature": pl.String, "psi": pl.Float64},
        )
        level_distribution = level_distributions(
            self.reference_levels, current_levels, level_config
        )
        monthly_distribution = pl.DataFrame(schema=MONTHLY_SCHEMA)
        if self.reference_time is not None and current_time is not None:
            monthly_distribution = monthly_level_distributions(
                self.reference_levels,
                self.reference_time,
                current_levels,
                current_time,
                level_config,
            )
        return PSIResult(
            psi=feature_scores,
            level_distribution=level_distribution,
            monthly_distribution=monthly_distribution,
            level_config=level_config,
        )
