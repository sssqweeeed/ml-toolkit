"""Reference/current and optional monthly level-share tables."""

from __future__ import annotations

import polars as pl

from ..feature_levels import LevelConfig

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
    """Tabulate every fitted level for both samples, including zero shares.

    Parameters
    ----------
    reference_levels : polars.DataFrame
        Nonempty transformed reference frame.
    current_levels : polars.DataFrame
        Nonempty transformed current frame.
    config : LevelConfig
        Fitted feature definitions matching the transformed columns.

    Returns
    -------
    polars.DataFrame
        feature, sample, level, rows, and share columns, ordered by configured
        feature, reference/current sample, and fitted level. Shares use each
        sample row count.
    """
    tables: list[pl.DataFrame] = []
    for feature_order, feature in enumerate(config.features):
        grid = pl.DataFrame(
            {"level": feature.levels, "level_order": range(len(feature.levels))},
            schema={"level": pl.String, "level_order": pl.Int64},
        )
        for sample_order, (sample, data) in enumerate(
            (("reference", reference_levels), ("current", current_levels))
        ):
            values = data.get_column(feature.name).cast(pl.String)
            counts = (
                pl.DataFrame({"level": values})
                .group_by("level")
                .agg(pl.len().alias("rows"))
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
        .cast(pl.Schema(LEVEL_SCHEMA))
    )


def monthly_level_distributions(
    reference_levels: pl.DataFrame,
    reference_time: pl.Series,
    current_levels: pl.DataFrame,
    current_time: pl.Series,
    config: LevelConfig,
) -> pl.DataFrame:
    """Tabulate fitted level shares in each observed sample month.

    Parameters
    ----------
    reference_levels : polars.DataFrame
        Transformed reference frame.
    reference_time : polars.Series
        Non-null Date/Datetime values aligned with reference rows.
    current_levels : polars.DataFrame
        Transformed current frame.
    current_time : polars.Series
        Non-null Date/Datetime values aligned with current rows.
    config : LevelConfig
        Fitted feature definitions matching both transformed frames.

    Returns
    -------
    polars.DataFrame
        feature, month, sample, level, and share columns. Months are
        first-of-month Dates; shares sum to one per feature, sample, and
        observed month.

    Notes
    -----
    Includes zero-share levels within observed months; does not create months
    with no rows.
    """
    tables: list[pl.DataFrame] = []
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
                {
                    "month": month,
                    "level": levels.get_column(feature.name).cast(pl.String),
                }
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
        .cast(pl.Schema(MONTHLY_SCHEMA))
    )
