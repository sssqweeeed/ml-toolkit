"""Numerical PSI contracts independent of fitted feature levels."""

import math

import polars as pl
import pytest

from ml_toolkit.psi import psi


def test_score_uses_union_of_levels_and_share_floor_without_renormalizing():
    reference = pl.Series(["a", "a", "b"])
    current = pl.Series(["a", "c", "c"])
    # Shares for a/b/c are (2/3, 1/3, .1) and (1/3, .1, 2/3).
    expected = (
        (1 / 3 - 2 / 3) * math.log(0.5)
        + (0.1 - 1 / 3) * math.log(0.3)
        + (2 / 3 - 0.1) * math.log((2 / 3) / 0.1)
    )
    assert psi(reference, current, epsilon=0.1) == pytest.approx(expected)
    assert psi(current, reference, epsilon=0.1) == pytest.approx(expected)


def test_equal_distributions_ignore_order_and_sample_size_and_count_nulls():
    reference = pl.Series(["a", None])
    current = pl.Series([None, "a", "a", None])
    assert psi(reference, current) == 0.0
    assert psi(reference, pl.Series(["a", "a"])) > 0.0


@pytest.mark.parametrize("epsilon", [0, -0.1, 1, float("nan"), float("inf")])
def test_score_rejects_invalid_share_floor(epsilon):
    values = pl.Series(["a"])
    with pytest.raises(ValueError, match="epsilon"):
        psi(values, values, epsilon=epsilon)


@pytest.mark.parametrize("empty_sample", ["reference", "current"])
def test_score_rejects_empty_samples(empty_sample):
    samples = {"reference": pl.Series(["a"]), "current": pl.Series(["a"])}
    samples[empty_sample] = pl.Series([], dtype=pl.String)
    with pytest.raises(ValueError, match="empty"):
        psi(**samples)
