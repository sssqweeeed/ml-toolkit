"""Structured PSI outputs; tables remain Polars DataFrames."""

from dataclasses import dataclass

import polars as pl

from .config import LevelConfig


@dataclass(frozen=True)
class PSIResult:
    psi: pl.DataFrame
    level_distribution: pl.DataFrame
    monthly_distribution: pl.DataFrame
    level_config: LevelConfig
