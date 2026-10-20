"""Reference/current and optional time-based level-share tables."""

from __future__ import annotations

import polars as pl

from .._time import paired_period_starts
from ..types import TimePeriod
from ..feature_levels import FeatureLevels

LEVEL_SCHEMA = {
    "feature": pl.String,
    "sample": pl.String,
    "level": pl.String,
    "rows": pl.Int64,
    "share": pl.Float64,
}
TIME_SCHEMA = {
    "feature": pl.String,
    "period_start": pl.Datetime("us"),
    "sample": pl.String,
    "level": pl.String,
    "share": pl.Float64,
}


def level_distributions(
    reference_levels: pl.DataFrame,
    current_levels: pl.DataFrame,
    feature_levels: tuple[FeatureLevels, ...],
) -> pl.DataFrame:
    tables: list[pl.DataFrame] = []
    for feature_order, feature in enumerate(feature_levels):
        grid = pl.DataFrame(
            {"level": feature.levels, "level_order": range(len(feature.levels))},
            schema={"level": pl.String, "level_order": pl.Int64},
        )
        for sample_order, (sample, data) in enumerate(
            (("reference", reference_levels), ("current", current_levels))
        ):
            values = data.get_column(feature.config.name).cast(pl.String)
            counts = (
                pl.DataFrame({"level": values})
                .group_by("level")
                .agg(pl.len().alias("rows"))
            )
            tables.append(
                grid.join(counts, on="level", how="left")
                .with_columns(
                    pl.col("rows").fill_null(0),
                    pl.lit(feature.config.name).alias("feature"),
                    pl.lit(sample).alias("sample"),
                    pl.lit(feature_order).alias("feature_order"),
                    pl.lit(sample_order).alias("sample_order"),
                )
                .with_columns((pl.col("rows") / len(data)).alias("share"))
            )
    return (
        pl.concat(tables)
        .sort("feature_order", "sample_order", "level_order")
        .select(*LEVEL_SCHEMA)
        .cast(pl.Schema(LEVEL_SCHEMA))
    )


def time_level_distributions(
    reference_levels: pl.DataFrame,
    reference_time: pl.Series,
    current_levels: pl.DataFrame,
    current_time: pl.Series,
    feature_levels: tuple[FeatureLevels, ...],
    period: TimePeriod = "month",
) -> pl.DataFrame:
    reference_start, current_start = paired_period_starts(
        reference_time, current_time, period
    )
    tables: list[pl.DataFrame] = []
    for feature_order, feature in enumerate(feature_levels):
        grid = pl.DataFrame(
            {"level": feature.levels, "level_order": range(len(feature.levels))},
            schema={"level": pl.String, "level_order": pl.Int64},
        )
        for sample_order, (sample, timestamp, levels) in enumerate(
            (
                ("reference", reference_start, reference_levels),
                ("current", current_start, current_levels),
            )
        ):
            period_start = timestamp
            values = pl.DataFrame(
                {
                    "period_start": period_start,
                    "level": levels.get_column(feature.config.name).cast(pl.String),
                }
            )
            counts = values.group_by("period_start", "level").agg(
                pl.len().alias("rows")
            )
            tables.append(
                values.select("period_start")
                .unique()
                .sort("period_start")
                .join(grid, how="cross")
                .join(counts, on=["period_start", "level"], how="left")
                .with_columns(pl.col("rows").fill_null(0))
                .with_columns(
                    (pl.col("rows") / pl.col("rows").sum().over("period_start")).alias(
                        "share"
                    ),
                    pl.lit(feature.config.name).alias("feature"),
                    pl.lit(sample).alias("sample"),
                    pl.lit(feature_order).alias("feature_order"),
                    pl.lit(sample_order).alias("sample_order"),
                )
            )
    return (
        pl.concat(tables)
        .sort("feature_order", "sample_order", "period_start", "level_order")
        .select(*TIME_SCHEMA)
        .cast(pl.Schema({**TIME_SCHEMA, "period_start": reference_start.dtype}))
    )
