"""Reference/current and optional monthly level-share tables."""

from __future__ import annotations

import polars as pl

from .config import LevelConfig


LEVEL_SCHEMA = {
    "feature": pl.String,
    "sample": pl.String,
    "level": pl.String,
    "rows": pl.Int64,
    "share": pl.Float64,
}
MONTHLY_SCHEMA = {
    "feature": pl.String,
    "month": pl.Date,
    "sample": pl.String,
    "level": pl.String,
    "share": pl.Float64,
}


def level_distributions(
    reference_levels: pl.DataFrame,
    current_levels: pl.DataFrame,
    config: LevelConfig,
) -> pl.DataFrame:
    """Return every fitted level for both samples, including zero-share levels."""
    tables = []
    for feature_order, feature in enumerate(config.features):
        grid = pl.DataFrame(
            {"level": feature.levels, "level_order": range(len(feature.levels))},
            schema={"level": pl.String, "level_order": pl.Int64},
        )
        for sample_order, (sample, data) in enumerate(
            (("reference", reference_levels), ("current", current_levels))
        ):
            values = data.get_column(feature.name).cast(pl.String)
            counts = pl.DataFrame({"level": values}).group_by("level").agg(
                pl.len().alias("rows")
            )
            tables.append(
                grid.join(counts, on="level", how="left")
                .with_columns(
                    pl.col("rows").fill_null(0),
                    pl.lit(feature.name).alias("feature"),
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
        .cast(LEVEL_SCHEMA)
    )


def monthly_level_distributions(
    reference_levels: pl.DataFrame,
    reference_time: pl.Series,
    current_levels: pl.DataFrame,
    current_time: pl.Series,
    config: LevelConfig,
) -> pl.DataFrame:
    """Return each level's share for every sample month, including zeros."""
    tables = []
    for feature_order, feature in enumerate(config.features):
        grid = pl.DataFrame(
            {"level": feature.levels, "level_order": range(len(feature.levels))},
            schema={"level": pl.String, "level_order": pl.Int64},
        )
        for sample_order, (sample, timestamp, levels) in enumerate(
            (
                ("reference", reference_time, reference_levels),
                ("current", current_time, current_levels),
            )
        ):
            month = timestamp.dt.truncate("1mo").cast(pl.Date).alias("month")
            values = pl.DataFrame(
                {"month": month, "level": levels.get_column(feature.name).cast(pl.String)}
            )
            counts = values.group_by("month", "level").agg(pl.len().alias("rows"))
            tables.append(
                values.select("month")
                .unique()
                .sort("month")
                .join(grid, how="cross")
                .join(counts, on=["month", "level"], how="left")
                .with_columns(pl.col("rows").fill_null(0))
                .with_columns(
                    (pl.col("rows") / pl.col("rows").sum().over("month")).alias(
                        "share"
                    ),
                    pl.lit(feature.name).alias("feature"),
                    pl.lit(sample).alias("sample"),
                    pl.lit(feature_order).alias("feature_order"),
                    pl.lit(sample_order).alias("sample_order"),
                )
            )
    return (
        pl.concat(tables)
        .sort("feature_order", "sample_order", "month", "level_order")
        .select(*MONTHLY_SCHEMA)
        .cast(MONTHLY_SCHEMA)
    )
