from abc import ABC, abstractmethod
from datetime import date
from typing import Any, Self

import numpy as np
import polars as pl


class InflationAdjuster(ABC):
    """
    Strategy for converting monetary values between price levels.

    The inflation module accepts only Polars Series with Date or
    Datetime dtype.

    factor(from_dates, to_dates) returns a multiplicative factor:

        value_at_to_date =
            value_at_from_date * factor(from_dates, to_dates)

    Both Series must have the same length. Broadcasting is deliberately
    handled outside the inflation module.

    Simple strategies can keep the default no-op fit().
    Learned inflation models can override fit().
    """

    def fit(
        self,
        dates: pl.Series,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> Self:
        _validate_dates(
            dates,
            name="dates",
        )
        return self

    @abstractmethod
    def factor(
        self,
        from_dates: pl.Series,
        to_dates: pl.Series,
    ) -> np.ndarray:
        raise NotImplementedError

    def adjust(
        self,
        values: np.ndarray,
        from_dates: pl.Series,
        to_dates: pl.Series,
    ) -> np.ndarray:
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

        return values * factor

    def to_config(self) -> dict[str, Any]:
        """Return an audit-friendly representation for experiment artifacts."""
        return {"strategy": type(self).__name__}


class ConstantRateInflation(InflationAdjuster):
    """
    Constant annual compound inflation.

    Example:
        annual_rate=0.08 means 8% annual inflation.
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
    ) -> np.ndarray:
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

    def to_config(self) -> dict[str, Any]:
        return {
            "strategy": type(self).__name__,
            "annual_rate": self.annual_rate,
        }


class ManualInflation(InflationAdjuster):
    """
    Sparse manually specified monthly price index.

    Example:
        ManualInflation({
            date(2025, 10, 1): 0.94,
            date(2025, 12, 1): 0.98,
            date(2026, 2, 1): 1.02,
            date(2026, 4, 1): 1.05,
        })

    Missing months between anchor points are filled using
    log-linear interpolation.

    Dates are interpreted at month granularity:
        2026-03-01
        2026-03-15
        2026-03-31

    all use the same March 2026 index.

    Extrapolation outside the supplied anchor range is forbidden.
    """

    def __init__(
        self,
        monthly_index: dict[date, float],
    ) -> None:
        if len(monthly_index) < 2:
            raise ValueError("At least two anchor months are required")

        normalized: dict[int, float] = {}

        for month, value in monthly_index.items():
            if not isinstance(month, date):
                raise TypeError("monthly_index keys must be datetime.date")

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
    ) -> np.ndarray:
        _validate_date_pair(
            from_dates=from_dates,
            to_dates=to_dates,
        )

        from_index = self._index_for_dates(from_dates)

        to_index = self._index_for_dates(to_dates)

        return to_index / from_index

    def to_config(self) -> dict[str, Any]:
        return {
            "strategy": type(self).__name__,
            "monthly_index": {
                self._format_month(month): float(np.exp(log_index))
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
    ) -> np.ndarray:
        months = self._to_month_numbers(dates)

        first_month = self._anchor_months[0]
        last_month = self._anchor_months[-1]

        outside_range = (months < first_month) | (months > last_month)

        if np.any(outside_range):
            missing = sorted(
                {self._format_month(month) for month in months[outside_range]}
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
    ) -> np.ndarray:
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


def inflation_from_config(config: dict[str, Any]) -> InflationAdjuster:
    """Reconstruct a supported inflation strategy from its persisted state."""
    strategy = config["strategy"]
    if strategy == "ConstantRateInflation":
        return ConstantRateInflation(config["annual_rate"])
    if strategy == "ManualInflation":
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
    if not isinstance(
        dates,
        pl.Series,
    ):
        raise TypeError(f"{name} must be polars.Series, got {type(dates).__name__}")

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
