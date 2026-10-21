"""Shared calendar grouping for diagnostic tables and figures."""

import polars as pl

from .types import TimePeriod

PERIOD_INTERVALS: dict[TimePeriod, str] = {
    "minute": "1m",
    "hour": "1h",
    "day": "1d",
    "week": "1w",
    "month": "1mo",
    "quarter": "1q",
    "year": "1y",
}


def period_starts(time: pl.Series, period: TimePeriod) -> pl.Series:
    if period not in PERIOD_INTERVALS:
        raise ValueError(f"Unsupported time period: {period}")
    if time.null_count():
        raise ValueError("timestamps must not contain missing values")
    dtype = time.dtype
    if dtype == pl.Date:
        if period in ("minute", "hour"):
            raise ValueError("Minute/hour grouping requires Datetime, not Date")
        time = time.cast(pl.Datetime("us"))
    elif isinstance(dtype, pl.Datetime):
        time = time.cast(pl.Datetime("us", time_zone=dtype.time_zone))
    else:
        raise TypeError("timestamps must have Date or Datetime dtype")
    return time.dt.truncate(PERIOD_INTERVALS[period]).alias("period_start")


def paired_period_starts(
    reference: pl.Series, current: pl.Series, period: TimePeriod
) -> tuple[pl.Series, pl.Series]:
    reference_start = period_starts(reference, period)
    current_start = period_starts(current, period)
    if reference_start.dtype != current_start.dtype:
        raise ValueError(
            "Reference and current timestamps must have matching time zones"
        )
    # Compare original instants, not rounded period starts. A split may fall
    # inside one period; shared endpoints are allowed, interleaved samples are not.
    ordered: bool | None = pl.select(
        pl.lit(reference).max() <= pl.lit(current).min()
    ).item()
    if ordered is False:
        raise ValueError(
            "Reference timestamps must precede or equal current timestamps: "
            "max(reference) <= min(current). "
            "Time ranges may share an endpoint but must not otherwise overlap."
        )
    return reference_start, current_start
