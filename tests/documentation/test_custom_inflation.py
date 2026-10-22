"""The documented project strategy preserves calendar and conversion semantics."""

from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest

from examples.custom_inflation import AnnualRateTable
from ml_toolkit.inflation import inflation_from_config

pytestmark = pytest.mark.documentation


def test_calendar_rates_compound_and_reverse_across_leap_year():
    strategy = AnnualRateTable({2024: 0.03, 2025: 0.07})
    start = pl.Series([date(2024, 1, 1), date(2024, 1, 1)])
    end = pl.Series([date(2024, 7, 2), date(2026, 1, 1)])
    factors = strategy.factor(start, end)
    np.testing.assert_allclose(factors, [1.03**0.5, 1.03 * 1.07])
    amounts = np.array([100.0, 200.0])
    changed = strategy.adjust(amounts, start, end)
    np.testing.assert_allclose(strategy.adjust(changed, end, start), amounts)


def test_fixed_policy_fit_keeps_rates_and_uses_local_calendar_dates():
    rates = {2024: 0.08}
    strategy = AnnualRateTable(rates)
    rates[2024] = 0.90
    zone = ZoneInfo("Asia/Tokyo")
    start = pl.Series([datetime(2024, 1, 1, 0, 30, tzinfo=zone)])
    end = pl.Series([datetime(2025, 1, 1, 0, 30, tzinfo=zone)])
    assert strategy.fit(start, np.array([100.0])) is strategy
    np.testing.assert_allclose(strategy.factor(start, end), [1.08])


@pytest.mark.parametrize(
    "rates",
    [{}, {2024: 0.03, 2026: 0.04}, {2024: -1.0}, {2024: float("nan")}],
)
def test_rejects_incomplete_or_invalid_policy(rates):
    with pytest.raises(ValueError):
        AnnualRateTable(rates)


def test_rejects_unsupported_dates_and_bundled_serialization():
    strategy = AnnualRateTable({2024: 0.03})
    good = pl.Series([date(2024, 1, 1)])
    with pytest.raises(ValueError, match="outside"):
        strategy.factor(good, pl.Series([date(2025, 1, 2)]))
    with pytest.raises(ValueError, match="null"):
        strategy.factor(good, pl.Series([None], dtype=pl.Date))
    with pytest.raises(ValueError, match="equal lengths"):
        strategy.factor(good, good.head(0))
    with pytest.raises(TypeError, match="Date or Datetime"):
        strategy.factor(good, pl.Series(["2024-06-01"]))
    with pytest.raises(ValueError, match="Unsupported inflation strategy"):
        inflation_from_config(strategy.to_config())
