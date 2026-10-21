import logging
from abc import ABC, abstractmethod
from datetime import date
from typing import NotRequired, Self, TypedDict

import numpy as np
import polars as pl

from .types import FloatArray, IntArray


class InflationConfig(TypedDict):
    strategy: str
    annual_rate: NotRequired[float]
    monthly_index: NotRequired[dict[str, float]]


logger = logging.getLogger("ml_toolkit.inflation")


class InflationAdjuster(ABC):
    """Convert monetary values between price levels.

    Notes
    -----
    The inflation module accepts only non-null Polars Series with Date or
    Datetime dtype. factor(from_dates, to_dates) returns multipliers such that
    value_at_to_date = value_at_from_date * factor(from_dates, to_dates). Both
    date Series must have equal length; date broadcasting belongs to the
    caller. Simple strategies can keep the default no-op fit method; learned
    inflation models can override it.
    """

    def fit(
        self,
        dates: pl.Series,
        y: FloatArray,
        sample_weight: FloatArray | None = None,
    ) -> Self:
        """Validate dates and return the strategy without learning parameters.

        Parameters
        ----------
        dates : polars.Series
            Non-null Date or Datetime observations.
        y : numpy.ndarray
            Targets available to learned subclasses; unused by the default
            implementation.
        sample_weight : numpy.ndarray or None, default None
            Optional row weights for learned subclasses; unused by the default
            implementation.

        Returns
        -------
        Self
            This strategy, unchanged.

        Raises
        ------
        TypeError
            dates is not a Polars Date/Datetime Series.
        ValueError
            dates contains nulls.
        """
        _validate_dates(
            dates,
            name="dates",
        )
        logger.debug(
            "Validated inflation strategy %s for %s dates",
            type(self).__name__,
            len(dates),
        )
        return self

    @abstractmethod
    def factor(
        self,
        from_dates: pl.Series,
        to_dates: pl.Series,
    ) -> FloatArray:
        """Return one multiplicative price-level conversion factor per date pair.

        Parameters
        ----------
        from_dates : polars.Series
            Non-null Date/Datetime values representing original price dates.
        to_dates : polars.Series
            Non-null Date/Datetime values representing desired price dates, with
            equal length to from_dates.

        Returns
        -------
        numpy.ndarray of shape (n_dates,)
            Factors converting values at from_dates to equivalent values at
            to_dates.
        """
        raise NotImplementedError

    def adjust(
        self,
        values: FloatArray,
        from_dates: pl.Series,
        to_dates: pl.Series,
    ) -> FloatArray:
        """Apply price-level factors to monetary value arrays.

        Parameters
        ----------
        values : numpy.ndarray
            Float values at from_dates. A zero-dimensional array is broadcast to
            all dates; other arrays must contain exactly one value per date.
        from_dates : polars.Series
            Non-null Date/Datetime values representing original price dates.
        to_dates : polars.Series
            Non-null Date/Datetime values representing desired price dates, with
            equal length to from_dates.

        Returns
        -------
        numpy.ndarray of shape (n_dates,)
            Values expressed at the corresponding to_dates price levels.

        Raises
        ------
        ValueError
            Date lengths differ, dates contain nulls, or values do not align with
            dates.
        """
        values = np.asarray(
            values,
            dtype=float,
        )

        factor = self.factor(
            from_dates=from_dates,
            to_dates=to_dates,
        )

        if values.ndim == 0:
            values = np.full(
                len(from_dates),
                float(values),
                dtype=float,
            )
        else:
            values = values.reshape(-1)

        if values.shape != (len(from_dates),):
            raise ValueError("values must contain exactly one value per date")

        logger.debug(
            "Adjusting inflation with %s: %s values", type(self).__name__, len(values)
        )
        return values * factor

    def to_config(self) -> InflationConfig:
        return {"strategy": type(self).__name__}


class ConstantRateInflation(InflationAdjuster):
    """Convert price levels using constant annual compound inflation.

    Parameters
    ----------
    annual_rate : float
        Annual proportional rate greater than -1; 0.08 means 8% annual
        inflation.

    Notes
    -----
    Elapsed years are computed using 365.25 days, including fractional days.

    Examples
    --------
    Express 100 monetary units at a later price level, then reverse the conversion.
    These small amounts illustrate the calculation, not an observed price index.

    >>> from datetime import date
    >>> import numpy as np
    >>> import polars as pl
    >>> from ml_toolkit.inflation import ConstantRateInflation
    >>> from_dates = pl.Series([date(2025, 1, 1)])
    >>> to_dates = pl.Series([date(2026, 1, 1)])
    >>> inflation = ConstantRateInflation(annual_rate=0.08)
    >>> later = inflation.adjust(np.array([100.0]), from_dates, to_dates)
    >>> later.round(2).tolist()
    [107.99]
    >>> inflation.adjust(later, to_dates, from_dates).round(2).tolist()
    [100.0]

    The forward value is slightly below 108 because these dates are 365 days
    apart and the annual compounding convention uses 365.25 days.
    """

    def __init__(
        self,
        annual_rate: float,
    ) -> None:
        if annual_rate <= -1.0:
            raise ValueError("annual_rate must be greater than -1")

        self.annual_rate = float(annual_rate)

    def factor(
        self,
        from_dates: pl.Series,
        to_dates: pl.Series,
    ) -> FloatArray:
        """Calculate compound factors using elapsed time in 365.25-day years.

        Parameters
        ----------
        from_dates : polars.Series
            Non-null Date/Datetime values representing original price dates.
        to_dates : polars.Series
            Non-null Date/Datetime values representing desired price dates, with
            equal length to from_dates.

        Returns
        -------
        numpy.ndarray of shape (n_dates,)
            (1 + annual_rate) raised to elapsed years for each date pair.

        Raises
        ------
        TypeError
            Dates are not Polars Date/Datetime Series.
        ValueError
            Dates contain nulls or have different lengths.
        """
        _validate_date_pair(
            from_dates=from_dates,
            to_dates=to_dates,
        )

        from_ns = from_dates.cast(pl.Datetime("ns"))
        to_ns = to_dates.cast(pl.Datetime("ns"))

        seconds = (to_ns - from_ns).dt.total_seconds(fractional=True)

        years = seconds.to_numpy() / (365.25 * 24 * 60 * 60)

        return np.power(
            1.0 + self.annual_rate,
            years,
        )

    def to_config(self) -> InflationConfig:
        return {
            "strategy": type(self).__name__,
            "annual_rate": self.annual_rate,
        }


class ManualInflation(InflationAdjuster):
    """Convert prices using a sparse monthly index with log-linear interpolation.

    Parameters
    ----------
    monthly_index : dict[datetime.date, float]
        At least two distinct anchor months mapped to finite, strictly positive
        index values. Keys are interpreted at month granularity.

    Raises
    ------
    ValueError
        Too few anchors are supplied, a month occurs twice, or an index value
        is nonpositive or non-finite.

    Notes
    -----
    Missing months between anchors are filled by linear interpolation in
    log(index), producing constant compound growth between anchors. Every day
    in a month uses that month's index. Extrapolation outside the supplied
    anchor range is forbidden.

    Examples
    --------
    >>> from datetime import date
    >>> import polars as pl
    >>> from ml_toolkit.inflation import ManualInflation
    >>> inflation = ManualInflation({date(2026, 1, 1): 1.0, date(2026, 3, 1): 1.21})
    >>> factor = inflation.factor(
    ...     pl.Series([date(2026, 1, 15)]),
    ...     pl.Series([date(2026, 2, 28)]),
    ... )
    >>> round(float(factor[0]), 2)
    1.1
    """

    def __init__(
        self,
        monthly_index: dict[date, float],
    ) -> None:
        if len(monthly_index) < 2:
            raise ValueError("At least two anchor months are required")

        normalized: dict[int, float] = {}

        for month, value in monthly_index.items():

            value = float(value)

            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    "Inflation index values must be finite and strictly positive"
                )

            month_number = month.year * 12 + month.month - 1

            if month_number in normalized:
                raise ValueError(f"Month {month:%Y-%m} is specified more than once")

            normalized[month_number] = value

        sorted_items = sorted(normalized.items())

        self._anchor_months = np.asarray(
            [month for month, _ in sorted_items],
            dtype=np.int64,
        )

        self._anchor_log_index = np.log(
            np.asarray(
                [value for _, value in sorted_items],
                dtype=float,
            )
        )

    def factor(
        self,
        from_dates: pl.Series,
        to_dates: pl.Series,
    ) -> FloatArray:
        """Calculate monthly destination-to-source price index ratios.

        Parameters
        ----------
        from_dates : polars.Series
            Non-null Date/Datetime values representing original price dates.
        to_dates : polars.Series
            Non-null Date/Datetime values representing desired price dates, with
            equal length to from_dates.

        Returns
        -------
        numpy.ndarray of shape (n_dates,)
            Interpolated destination index divided by source index.

        Raises
        ------
        ValueError
            Date lengths differ, dates contain nulls, or a month is outside the
            anchor range.
        """
        _validate_date_pair(
            from_dates=from_dates,
            to_dates=to_dates,
        )

        from_index = self._index_for_dates(from_dates)

        to_index = self._index_for_dates(to_dates)

        return to_index / from_index

    def to_config(self) -> InflationConfig:
        return {
            "strategy": type(self).__name__,
            "monthly_index": {
                self._format_month(int(month)): float(np.exp(log_index))
                for month, log_index in zip(
                    self._anchor_months,
                    self._anchor_log_index,
                    strict=True,
                )
            },
        }

    def _index_for_dates(
        self,
        dates: pl.Series,
    ) -> FloatArray:
        months = self._to_month_numbers(dates)

        first_month = self._anchor_months[0]
        last_month = self._anchor_months[-1]

        outside_range = (months < first_month) | (months > last_month)

        if np.any(outside_range):
            missing = sorted(
                {self._format_month(int(month)) for month in months[outside_range]}
            )

            raise ValueError(
                "Inflation index is not defined "
                f"for months: {missing}. "
                "Supported range: "
                f"{self._format_month(first_month)} - "
                f"{self._format_month(last_month)}"
            )

        # Linear interpolation in log(index):
        #
        # log(I_t) is linear between anchor points,
        # so I_t grows geometrically / compound-wise.
        log_index = np.interp(
            months,
            self._anchor_months,
            self._anchor_log_index,
        )

        return np.exp(log_index)

    @staticmethod
    def _to_month_numbers(
        dates: pl.Series,
    ) -> IntArray:
        years = dates.dt.year().to_numpy().astype(np.int64)

        months = dates.dt.month().to_numpy().astype(np.int64)

        return years * 12 + months - 1

    @staticmethod
    def _format_month(
        month_number: int,
    ) -> str:
        year, month = divmod(
            int(month_number),
            12,
        )

        return f"{year:04d}-{month + 1:02d}"


def inflation_from_config(config: InflationConfig) -> InflationAdjuster:
    strategy = config["strategy"]
    if strategy == "ConstantRateInflation":
        if "annual_rate" not in config:
            raise KeyError("annual_rate")
        return ConstantRateInflation(config["annual_rate"])
    if strategy == "ManualInflation":
        if "monthly_index" not in config:
            raise KeyError("monthly_index")
        return ManualInflation(
            {
                date.fromisoformat(f"{month}-01"): value
                for month, value in config["monthly_index"].items()
            }
        )
    raise ValueError(f"Unsupported inflation strategy for serialization: {strategy}")


def _validate_dates(
    dates: pl.Series,
    *,
    name: str,
) -> None:

    dtype = dates.dtype

    is_date = dtype == pl.Date
    is_datetime = dtype.base_type() == pl.Datetime

    if not (is_date or is_datetime):
        raise TypeError(f"{name} must have Date or Datetime dtype, got {dtype}")

    if dates.null_count() > 0:
        raise ValueError(f"{name} must not contain null values")


def _validate_date_pair(
    *,
    from_dates: pl.Series,
    to_dates: pl.Series,
) -> None:
    _validate_dates(
        from_dates,
        name="from_dates",
    )
    _validate_dates(
        to_dates,
        name="to_dates",
    )

    if len(from_dates) != len(to_dates):
        raise ValueError("from_dates and to_dates must have equal length")
