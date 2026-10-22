"""Calendar-year conversion policies supplied by a project."""

from datetime import date
from math import isfinite, log1p

import numpy as np
import polars as pl

from ml_toolkit.inflation import InflationAdjuster
from ml_toolkit.types import FloatArray


class AnnualRateTable(InflationAdjuster):
    """Compound a different supplied rate in each calendar year.

    Parameters
    ----------
    annual_rates : dict[int, float]
        Consecutive calendar years mapped to finite rates greater than -1.
        A rate of 0.08 represents 8% growth over that complete year.

    Notes
    -----
    Partial years use elapsed calendar days divided by days in that year.
    Datetimes use their local calendar date. The range includes January 1
    after the final configured year. Rates are fixed policy inputs; inherited
    fit validates dates and does not estimate parameters from targets.
    """

    def __init__(self, annual_rates: dict[int, float]) -> None:
        years = sorted(annual_rates)
        if not years or years != list(range(years[0], years[-1] + 1)):
            raise ValueError("Supply consecutive calendar years")
        if years[0] < 1 or years[-1] > 9998:
            raise ValueError("Years must be between 1 and 9998")
        if any(not isfinite(rate) or rate <= -1 for rate in annual_rates.values()):
            raise ValueError("Annual rates must be finite and greater than -1")
        self.annual_rates = annual_rates.copy()
        self._start = date(years[0], 1, 1)
        self._end = date(years[-1] + 1, 1, 1)

    def factor(self, from_dates: pl.Series, to_dates: pl.Series) -> FloatArray:
        """Return price-level ratios for equally sized, non-null date series."""
        if len(from_dates) != len(to_dates):
            raise ValueError("Date series must have equal lengths")
        factors = np.exp(self._log_level(to_dates) - self._log_level(from_dates))
        if not np.isfinite(factors).all() or (factors <= 0).any():
            raise ValueError("Conversion factors must be finite and positive")
        return factors

    def _log_level(self, dates: pl.Series) -> FloatArray:
        if dates.dtype != pl.Date and dates.dtype.base_type() != pl.Datetime:
            raise TypeError("Dates must have Date or Datetime dtype")
        if dates.null_count():
            raise ValueError("Dates must not contain nulls")
        calendar_dates = dates.dt.date()
        if ((calendar_dates < self._start) | (calendar_dates > self._end)).any():
            raise ValueError("Dates are outside the configured years")
        levels = np.zeros(len(dates), dtype=float)
        for year, rate in self.annual_rates.items():
            start, end = date(year, 1, 1), date(year + 1, 1, 1)
            elapsed = (calendar_dates - start).dt.total_days().to_numpy()
            fraction = np.clip(elapsed / (end - start).days, 0.0, 1.0)
            levels += log1p(rate) * fraction
        return levels
