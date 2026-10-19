"""PSI scores and distribution tables as Polars DataFrames."""

from dataclasses import dataclass

import polars as pl

from ..feature_levels import LevelConfig


@dataclass(frozen=True)
class PSIResult:
    """Store PSI scores and distribution tables as Polars DataFrames.

    Attributes
    ----------
    psi : polars.DataFrame
        One row per feature with feature (String) and psi (Float64) columns.
    level_distribution : polars.DataFrame
        Columns feature, sample, and level (String), rows (Int64), and share
        (Float64). Samples are reference and current; every fitted level is
        present, including zero-share levels.
    time_distribution : polars.DataFrame
        Columns feature, sample, and level (String), period_start (Datetime), and share
        (Float64). Contains every fitted level in each observed sample period;
        preserves the input timezone. Empty with naive Datetime when time was omitted.
    level_config : LevelConfig
        Reference-fitted categories, interval boundaries, and special level
        definitions.
    """

    psi: pl.DataFrame
    level_distribution: pl.DataFrame
    time_distribution: pl.DataFrame
    level_config: LevelConfig
