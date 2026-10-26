from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from datetime import date
from typing import TYPE_CHECKING, Literal, NotRequired, Self, TypedDict

import numpy as np
import polars as pl

from ._console import Logger
from .types import FloatArray, IntArray

if TYPE_CHECKING:
    from scipy.interpolate import BSpline

    from .data import Pool


Extrapolation = Literal["linear", "constant", "error", "damped"]


class LearnedInflationState(TypedDict):
    """JSON representation of a fitted temporal effect, independent of features."""

    version: int
    origin_ns: int
    upper: float
    coefficients: list[float]
    knots: list[float]


class _LearnedOptions(TypedDict):
    alpha: float
    control_features: list[str] | None
    extrapolation: Extrapolation
    damping_half_life_years: float


class InflationConfig(TypedDict):
    strategy: str
    annual_rate: NotRequired[float]
    monthly_index: NotRequired[dict[str, float]]
    alpha: NotRequired[float]
    control_features: NotRequired[list[str] | None]
    extrapolation: NotRequired[Extrapolation]
    damping_half_life_years: NotRequired[float]
    n_knots: NotRequired[int]
    fitted_state: NotRequired[LearnedInflationState]


logger = Logger("ml_toolkit.inflation")


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
        data: Pool,
    ) -> Self:
        """Validate dates and return the strategy without learning parameters.

        Parameters
        ----------
        data : ml_toolkit.data.Pool
            Training observations. The default implementation validates only
            data.time; learned subclasses can also use targets, weights and X.

        Returns
        -------
        Self
            This strategy, unchanged.

        Raises
        ------
        TypeError
            data.time is not a Polars Date/Datetime Series.
        ValueError
            data.time contains nulls.
        """
        _validate_dates(
            data.time,
            name="dates",
        )
        logger.debug(
            "Validated inflation strategy %s for %s dates",
            type(self).__name__,
            len(data),
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


class _LearnedInflation(InflationAdjuster):
    """Shared Gamma estimation and date-only inference for a separable index."""

    def __init__(
        self,
        *,
        alpha: float = 0.001,
        control_features: Sequence[str] | None = None,
        extrapolation: Extrapolation = "linear",
        damping_half_life_years: float = 1.0,
    ) -> None:
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        if extrapolation not in ("linear", "constant", "error", "damped"):
            raise ValueError(f"Unsupported extrapolation: {extrapolation}")
        if not np.isfinite(damping_half_life_years) or damping_half_life_years <= 0:
            raise ValueError("damping_half_life_years must be finite and positive")
        if isinstance(control_features, str):
            raise TypeError("control_features must be a sequence of feature names")
        names = None if control_features is None else tuple(control_features)
        if names is not None and len(set(names)) != len(names):
            raise ValueError("control_features must not contain duplicates")
        self.alpha = float(alpha)
        self.control_features = names
        self.extrapolation: Extrapolation = extrapolation
        self.damping_half_life_years = float(damping_half_life_years)
        self._n_knots: int | None = None
        self._state: LearnedInflationState | None = None
        self._spline: BSpline | None = None

    def fit(self, data: Pool) -> Self:
        """Estimate a common price trend using training targets and controls only.

        Parameters
        ----------
        data : ml_toolkit.data.Pool
            Finite positive targets, finite nonnegative weights with positive
            total, and non-null dates. Zero-weight rows are excluded before
            fitting any transformations or determining the training range.

        Returns
        -------
        Self
            Strategy with a newly fitted temporal effect.

        Notes
        -----
        The Gamma mean has log-link intercept + linear numeric effects +
        categorical effects + trend(time). There are no feature/time
        interactions: all objects share the same proportional price changes.
        Numeric missing values use training medians; all-missing numeric columns
        are dropped. Feature transformations are discarded after estimation.
        """
        from scipy.sparse import csr_matrix, hstack
        from sklearn.linear_model import GammaRegressor
        from sklearn.preprocessing import (
            OneHotEncoder,
            SplineTransformer,
            StandardScaler,
        )

        super().fit(data)
        target = data.y.to_numpy().astype(float)
        weight = data.weight.to_numpy().astype(float)
        if not np.isfinite(target).all() or (target <= 0).any():
            raise ValueError(
                "Learned inflation targets must be finite and strictly positive"
            )
        if (
            not np.isfinite(weight).all()
            or (weight < 0).any()
            or not np.isfinite(weight.sum())
            or weight.sum() <= 0
        ):
            raise ValueError(
                "Weights must be finite, nonnegative, and have positive total"
            )
        names = (
            data.X.columns
            if self.control_features is None
            else list(self.control_features)
        )
        missing = sorted(set(names) - set(data.X.columns))
        if missing:
            raise ValueError(f"Control features are missing from X: {missing}")
        active = weight > 0
        timestamps = _timestamps_ns(data.time)[active]
        minimum_dates = 2 if self._n_knots is None else self._n_knots
        if len(np.unique(timestamps)) < minimum_dates:
            raise ValueError(
                f"At least {minimum_dates} distinct positive-weight dates are required"
            )
        origin = int(timestamps.min())
        years = _years_from_origin(timestamps, origin)
        upper = float(years.max())
        knots: list[float] = []
        if self._n_knots is None:
            temporal = years[:, None]
        else:
            transformer = SplineTransformer(
                n_knots=self._n_knots, degree=3, knots="uniform", include_bias=False
            )
            temporal = transformer.fit_transform(years[:, None])
            knots = transformer.bsplines_[0].t.tolist()
        blocks = [csr_matrix(temporal)]
        numeric: list[FloatArray] = []
        categorical: list[str] = []
        features = data.X.filter(pl.Series(active))
        for name in names:
            if name in data.cat_features:
                categorical.append(name)
                continue
            column = features[name]
            if not (
                column.dtype.is_numeric()
                or column.dtype == pl.Boolean
                or column.dtype == pl.Null
            ):
                raise TypeError(
                    f"Control feature {name!r} must be numeric or listed in cat_features"
                )
            values = column.cast(pl.Float64).to_numpy().copy()
            if np.isinf(values).any():
                raise ValueError(f"Control feature {name!r} must not contain infinity")
            if np.isnan(values).all():
                continue
            values[np.isnan(values)] = float(np.nanmedian(values))
            numeric.append(values)
        if numeric:
            scaled = StandardScaler().fit_transform(
                np.column_stack(numeric), sample_weight=weight[active]
            )
            blocks.append(csr_matrix(scaled))
        if categorical:
            categories = (
                features.select(categorical)
                .with_columns(pl.all().cast(pl.String).fill_null("unknown"))
                .to_numpy()
            )
            blocks.append(OneHotEncoder(sparse_output=True).fit_transform(categories))
        regressor = GammaRegressor(alpha=self.alpha, max_iter=1000, tol=1e-8)
        regressor.fit(
            hstack(blocks, format="csr"), target[active], sample_weight=weight[active]
        )
        coefficients = regressor.coef_[: temporal.shape[1]].tolist()
        if self._n_knots is not None:
            # include_bias=False dropped the final basis column.
            coefficients.append(0.0)
        state: LearnedInflationState = {
            "version": 1,
            "origin_ns": origin,
            "upper": upper,
            "coefficients": coefficients,
            "knots": knots,
        }
        self.restore_state(state)
        logger.debug(
            "Fitted %s with %s positive-weight rows", type(self).__name__, len(years)
        )
        return self

    def factor(self, from_dates: pl.Series, to_dates: pl.Series) -> FloatArray:
        """Return common destination/source index ratios without object features.

        Parameters
        ----------
        from_dates, to_dates : polars.Series
            Equal-length non-null Date/Datetime series. Requested dates may be
            outside training according to the configured extrapolation policy.

        Returns
        -------
        numpy.ndarray
            Finite, strictly positive multipliers, one per aligned date pair.

        Raises
        ------
        RuntimeError
            The strategy has not been fitted.
        ValueError
            Date inputs are invalid, extrapolation is forbidden, or the requested
            conversion overflows or underflows floating-point representation.
        """
        _validate_date_pair(from_dates=from_dates, to_dates=to_dates)
        difference = self._log_level(to_dates) - self._log_level(from_dates)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            result = np.exp(difference)
        if not np.isfinite(result).all() or (result <= 0).any():
            raise ValueError("Inflation factors must be finite and strictly positive")
        return result

    def to_config(self) -> InflationConfig:
        """Return JSON-compatible constructor settings and optional fitted state."""
        config: InflationConfig = {
            "strategy": type(self).__name__,
            "alpha": self.alpha,
            "control_features": (
                None if self.control_features is None else list(self.control_features)
            ),
            "extrapolation": self.extrapolation,
            "damping_half_life_years": self.damping_half_life_years,
        }
        if self._n_knots is not None:
            config["n_knots"] = self._n_knots
        if self._state is not None:
            config["fitted_state"] = deepcopy(self._state)
        return config

    def restore_state(self, state: LearnedInflationState) -> None:
        """Validate and restore the temporal state saved by to_config()."""
        from scipy.interpolate import BSpline

        coefficients = np.asarray(state["coefficients"], dtype=float)
        knots = np.asarray(state["knots"], dtype=float)
        upper = state["upper"]
        valid = (
            state["version"] == 1
            and type(state["origin_ns"]) is int
            and np.iinfo(np.int64).min <= state["origin_ns"] <= np.iinfo(np.int64).max
            and np.isfinite(upper)
            and upper > 0
            and coefficients.ndim == knots.ndim == 1
            and np.isfinite(coefficients).all()
            and np.isfinite(knots).all()
        )
        spline: BSpline | None = None
        if self._n_knots is None:
            valid = valid and coefficients.shape == (1,) and knots.size == 0
        else:
            valid = (
                valid
                and knots.shape == (self._n_knots + 6,)
                and coefficients.shape == (self._n_knots + 2,)
                and np.all(np.diff(knots) > 0)
            )
            if valid:
                valid = knots[3] == 0.0 and knots[-4] == upper
                spline = BSpline(knots, coefficients, 3, extrapolate=False)
        if not valid:
            raise ValueError("Invalid or unsupported learned inflation state")
        self._state = deepcopy(state)
        self._spline = spline

    def _log_level(self, dates: pl.Series) -> FloatArray:
        state = self._state
        if state is None:
            raise RuntimeError("Inflation strategy has not been fitted")
        years = _years_from_origin(_timestamps_ns(dates), state["origin_ns"])
        upper = state["upper"]
        outside = (years < 0) | (years > upper)
        if self.extrapolation == "error" and outside.any():
            raise ValueError("Dates are outside the fitted inflation range")
        clipped = np.clip(years, 0.0, upper)
        if self._spline is None:
            slope = state["coefficients"][0]
            level = clipped * slope
            slopes = np.full(len(years), slope, dtype=float)
        else:
            level = np.asarray(self._spline(clipped), dtype=float)
            slopes = np.asarray(self._spline(clipped, nu=1), dtype=float)
        if self.extrapolation in ("linear", "damped"):
            distance = years - clipped
            if self.extrapolation == "damped":
                future = distance > 0
                tau = self.damping_half_life_years / np.log(2.0)
                distance[future] = -tau * np.expm1(-distance[future] / tau)
            level += slopes * distance
        return level


class LogLinearInflation(_LearnedInflation):
    """Learn a constant compound trend while controlling for object composition.

    Parameters
    ----------
    alpha : float, default 0.001
        Nonnegative L2 regularization of both temporal and control effects.
    control_features : sequence of str or None, default None
        Controls from Pool.X. None uses every available column; an empty
        sequence fits a date-only model. Categories follow Pool.cat_features.
    extrapolation : {'linear', 'constant', 'error', 'damped'}, default 'linear'
        Outside training, continue the boundary log-index slope, hold the level,
        reject the date, or exponentially damp the future slope toward zero.
        Damped extrapolation before the first training date remains linear.
    damping_half_life_years : float, default 1.0
        Positive number of years for the extrapolated future slope to halve.

    Notes
    -----
    Fits a Gamma mean with log-link to positive targets, with linear numeric
    and one-hot categorical controls. The inferred temporal effect is a common
    index, not an object-specific or causal inflation estimate. Missing controls
    and time-correlated characteristics can limit its interpretation. Dates use
    elapsed time in 365.25-day years; timezone-aware datetimes use UTC instants.
    At least two distinct dates with positive weight are required.
    """


class SplineInflation(_LearnedInflation):
    """Learn a changing common price trend with a regularized cubic B-spline.

    Parameters
    ----------
    n_knots : int, default 5
        Number of uniformly spaced knots over positive-weight training dates.
        Must be at least two; at least n_knots distinct dates are required.
    alpha : float, default 0.001
        Nonnegative L2 regularization of temporal basis and control effects.
    control_features : sequence of str or None, default None
        Controls from Pool.X; None uses every column and an empty sequence uses
        dates only. Numeric effects are linear, categories are one-hot encoded.
    extrapolation : {'linear', 'constant', 'error', 'damped'}, default 'linear'
        Same boundary policy as LogLinearInflation. The spline polynomial is
        never continued outside training; linear uses its boundary tangent.
    damping_half_life_years : float, default 1.0
        Positive number of years for the extrapolated future slope to halve.

    Notes
    -----
    Uses the same weighted Gamma mean and shared-index assumption as
    LogLinearInflation. Only time is spline-transformed; feature/time
    interactions, separate group trends and seasonality are not modeled.
    """

    def __init__(
        self,
        *,
        n_knots: int = 5,
        alpha: float = 0.001,
        control_features: Sequence[str] | None = None,
        extrapolation: Extrapolation = "linear",
        damping_half_life_years: float = 1.0,
    ) -> None:
        super().__init__(
            alpha=alpha,
            control_features=control_features,
            extrapolation=extrapolation,
            damping_half_life_years=damping_half_life_years,
        )
        if type(n_knots) is not int or n_knots < 2:
            raise ValueError("n_knots must be an integer of at least two")
        self._n_knots = n_knots


def _timestamps_ns(dates: pl.Series) -> IntArray:
    return dates.cast(pl.Datetime("ns")).cast(pl.Int64).to_numpy()


def _years_from_origin(timestamps: IntArray, origin: int) -> FloatArray:
    # Difference seconds and nanosecond remainders separately: subtracting two
    # valid int64 nanosecond timestamps can itself overflow across centuries.
    seconds, nanos = np.divmod(timestamps, 1_000_000_000)
    origin_seconds, origin_nanos = divmod(origin, 1_000_000_000)
    elapsed = (seconds - origin_seconds) + (nanos - origin_nanos) / 1e9
    return elapsed / (365.25 * 24 * 60 * 60)


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
    if strategy in ("LogLinearInflation", "SplineInflation"):
        options: _LearnedOptions = {
            "alpha": config.get("alpha", 0.001),
            "control_features": config.get("control_features"),
            "extrapolation": config.get("extrapolation", "linear"),
            "damping_half_life_years": config.get("damping_half_life_years", 1.0),
        }
        learned = (
            LogLinearInflation(**options)
            if strategy == "LogLinearInflation"
            else SplineInflation(n_knots=config.get("n_knots", 5), **options)
        )
        if "fitted_state" in config:
            learned.restore_state(config["fitted_state"])
        return learned
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
