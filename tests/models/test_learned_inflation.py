"""Statistical behavior, extrapolation and persistence of common price indices."""

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from ml_toolkit.data import Pool
from ml_toolkit.inflation import (
    ConstantRateInflation,
    LogLinearInflation,
    ManualInflation,
    SplineInflation,
    inflation_from_config,
)
from ml_toolkit.models import CatBoostRegressionModel, InflationAdjustedModel
from tests.models.support import PARAMS

STRATEGIES = [LogLinearInflation, SplineInflation]
FAMILIES = ["Gamma", "Poisson"]
ORIGIN = datetime(2020, 1, 1)
YEAR = 365.25 * 86400


def dates(years):
    return pl.Series(
        "time", [ORIGIN + timedelta(seconds=float(t * YEAR)) for t in years]
    )


def observations(trend=lambda t: 0.08 * t):
    rng = np.random.default_rng(42)
    years = np.repeat(np.linspace(0, 4, 49), 12)
    numeric = rng.normal(size=len(years))
    # More expensive objects become common over time, independently of inflation.
    premium = rng.random(len(years)) < (0.1 + 0.8 * years / 4)
    target = np.exp(4 + 0.3 * numeric + 1.2 * premium + trend(years))
    return Pool(
        pl.DataFrame(
            {"size": numeric, "category": np.where(premium, "premium", "budget")}
        ),
        pl.Series("target", target),
        dates(years),
        pl.Series("weight", np.linspace(0.5, 1.5, len(years))),
        cat_features=("category",),
        numerical_features=("size",),
    )


@pytest.mark.parametrize("factory", STRATEGIES)
@pytest.mark.parametrize("rate", [-0.1, 0.0, 0.08])
def test_recovers_growth_flat_prices_and_deflation_despite_composition(factory, rate):
    data = observations(lambda t: rate * t)
    strategy = factory(alpha=1e-8).fit(data)
    a, b = dates([0, 1]), dates([2, 4])
    np.testing.assert_allclose(
        strategy.factor(a, b), np.exp(rate * np.array([2, 3])), rtol=0.002
    )
    if rate == 0.0:
        uncontrolled = factory(alpha=1e-8, control_features=[]).fit(data)
        assert uncontrolled.factor(dates([0]), dates([4]))[0] > 1.5


def test_spline_recovers_changing_tempo_better_than_constant_trend():
    curve = lambda t: 0.02 * t + 0.035 * t**2
    data = observations(curve)
    a, b = dates([0.0, 1.0, 2.0, 3.0]), dates([0.8, 1.8, 2.8, 3.8])
    expected = np.exp(curve(np.array([0.8, 1.8, 2.8, 3.8])) - curve(np.arange(4)))
    spline = SplineInflation(alpha=1e-8).fit(data)
    linear = LogLinearInflation(alpha=1e-8).fit(data)
    np.testing.assert_allclose(spline.factor(a, b), expected, rtol=0.003)
    assert (
        np.mean(abs(spline.factor(a, b) - expected))
        < np.mean(abs(linear.factor(a, b) - expected)) / 10
    )


@pytest.mark.parametrize("factory", STRATEGIES)
def test_poisson_recovers_frequency_trend_with_zeros_and_varying_exposure(factory):
    rng = np.random.default_rng(42)
    years = np.repeat(np.linspace(0, 4, 49), 200)
    numeric = rng.normal(size=len(years))
    premium = rng.random(len(years)) < (0.1 + 0.8 * years / 4)
    exposure = rng.uniform(0.25, 2, len(years))
    rate = np.exp(-0.5 + 0.3 * numeric + 0.8 * premium + 0.2 * years)
    count = rng.poisson(exposure * rate)
    data = Pool(
        pl.DataFrame({"size": numeric, "category": premium.astype(str)}),
        pl.Series("frequency", count / exposure),
        dates(years),
        pl.Series("exposure", exposure),
        cat_features=("category",),
        numerical_features=("size",),
    )
    assert (data.y == 0).any()
    assert (data.y != data.y.floor()).any()
    strategy = factory(family="Poisson", alpha=1e-8).fit(data)
    np.testing.assert_allclose(
        strategy.factor(dates([0, 1]), dates([2, 4])),
        np.exp(0.2 * np.array([2, 3])),
        rtol=0.08,
    )


@pytest.mark.parametrize("factory", STRATEGIES)
@pytest.mark.parametrize("family", FAMILIES)
def test_factors_are_reversible_transitive_and_accept_empty_inputs(factory, family):
    strategy = factory(family=family).fit(observations())
    a, b, c = dates([-1, 1]), dates([2, 3]), dates([4, 6])
    np.testing.assert_array_equal(strategy.factor(a, a), np.ones(2))
    np.testing.assert_allclose(strategy.factor(a, b) * strategy.factor(b, a), 1)
    np.testing.assert_allclose(
        strategy.factor(a, b) * strategy.factor(b, c), strategy.factor(a, c)
    )
    values = np.array([100.0, 200.0])
    np.testing.assert_allclose(
        strategy.adjust(strategy.adjust(values, a, b), b, a), values
    )
    assert strategy.factor(a.head(0), a.head(0)).shape == (0,)


@pytest.mark.parametrize("factory", STRATEGIES)
@pytest.mark.parametrize("family", FAMILIES)
def test_all_extrapolation_policies_and_boundary_continuity(factory, family):
    data = observations()
    linear = factory(family=family, alpha=1e-8).fit(data)
    # Estimate the boundary derivative from inside the range, independently of state.
    eps = 1e-5
    slope = np.log(linear.factor(dates([4 - eps]), dates([4]))[0]) / eps
    state = linear.to_config()
    for policy in ("linear", "constant", "damped", "error"):
        configured = {**state, "extrapolation": policy, "damping_half_life_years": 0.5}
        strategy = inflation_from_config(configured)
        np.testing.assert_allclose(strategy.factor(dates([0, 4]), dates([0, 4])), 1)
        if policy == "error":
            with pytest.raises(ValueError, match="outside"):
                strategy.factor(dates([4]), dates([4.1]))
            with pytest.raises(ValueError, match="outside"):
                strategy.factor(dates([-0.1]), dates([0]))
            continue
        distance = 1.0
        if policy == "constant":
            distance = 0.0
            np.testing.assert_allclose(strategy.factor(dates([-2]), dates([0])), 1)
        if policy == "damped":
            tau = 0.5 / np.log(2)
            distance = tau * (1 - np.exp(-1 / tau))
            np.testing.assert_allclose(
                strategy.factor(dates([-1]), dates([0])),
                linear.factor(dates([-1]), dates([0])),
            )
        np.testing.assert_allclose(
            strategy.factor(dates([4]), dates([5])), np.exp(slope * distance), rtol=2e-5
        )


@pytest.mark.parametrize("factory", STRATEGIES)
@pytest.mark.parametrize("family", FAMILIES)
def test_numeric_imputation_dropping_empty_columns_and_zero_weight_rows(factory, family):
    data = observations()
    if family == "Poisson":
        data = data.with_target(data.y / 100)
    # The control has no effect on y, but must be safely processed.
    extra = np.full(len(data), np.nan)
    extra[::2] = 2.0
    data = replace(
        data,
        X=data.X.with_columns(pl.Series("partial", extra), pl.lit(None).alias("empty")),
    )
    expected = factory(family=family).fit(data)
    outlier = replace(
        data.take([0]),
        X=data.X.head(1).with_columns(pl.lit(float("inf")).alias("partial")),
        y=pl.Series([1e100]),
        time=dates([-100]),
        weight=pl.Series([0.0]),
    )
    combined = replace(
        data,
        X=pl.concat([outlier.X, data.X]),
        y=pl.concat([outlier.y, data.y]),
        time=pl.concat([outlier.time, data.time]),
        weight=pl.concat([outlier.weight, data.weight]),
    )
    actual = factory(family=family).fit(combined)
    assert actual.to_config() == expected.to_config()
    without_extra = factory(family=family, control_features=["size", "category"]).fit(data)
    np.testing.assert_allclose(
        actual.factor(dates([0]), dates([5])),
        without_extra.factor(dates([0]), dates([5])),
        rtol=1e-5,
    )


@pytest.mark.parametrize("factory", STRATEGIES)
@pytest.mark.parametrize("family", FAMILIES)
def test_weights_match_row_replication_and_scaling(factory, family):
    data = observations()
    if family == "Poisson":
        data = data.with_target(data.y / 100)
    weight = np.tile([1, 2, 3], len(data) // 3)
    weighted = replace(data, weight=pl.Series(weight))
    repeated = data.take(np.repeat(np.arange(len(data)), weight))
    repeated = replace(repeated, weight=pl.Series(np.ones(len(repeated))))
    scaled = replace(weighted, weight=weighted.weight * 10)
    a, b = dates([0, 1]), dates([3, 5])
    fitted = factory(family=family).fit(weighted)
    np.testing.assert_allclose(
        fitted.factor(a, b), factory(family=family).fit(repeated).factor(a, b), rtol=2e-5
    )
    np.testing.assert_allclose(
        fitted.factor(a, b), factory(family=family).fit(scaled).factor(a, b), rtol=2e-5
    )


@pytest.mark.parametrize("factory", STRATEGIES)
@pytest.mark.parametrize("family", FAMILIES)
def test_json_roundtrip_needs_no_feature_transformations_or_fitting(
    factory, family, monkeypatch
):
    from sklearn.linear_model import GammaRegressor, PoissonRegressor
    from sklearn.preprocessing import SplineTransformer

    strategy = factory(family=family, control_features=["category"], extrapolation="damped").fit(
        observations()
    )
    config = json.loads(json.dumps(strategy.to_config(), allow_nan=False))

    def forbidden(*args, **kwargs):
        raise AssertionError("Restoring a strategy must not fit an estimator")

    monkeypatch.setattr(GammaRegressor, "fit", forbidden)
    monkeypatch.setattr(PoissonRegressor, "fit", forbidden)
    monkeypatch.setattr(SplineTransformer, "fit", forbidden)
    loaded = inflation_from_config(config)
    assert loaded.to_config()["family"] == family
    a, b = dates([-1, 1]), dates([3, 10])
    np.testing.assert_array_equal(loaded.factor(a, b), strategy.factor(a, b))
    config["fitted_state"]["coefficients"][0] = 999
    np.testing.assert_array_equal(loaded.factor(a, b), strategy.factor(a, b))
    exported = strategy.to_config()
    exported["fitted_state"]["coefficients"][0] = 999
    np.testing.assert_array_equal(loaded.factor(a, b), strategy.factor(a, b))


@pytest.mark.parametrize("factory", STRATEGIES)
@pytest.mark.parametrize("family", FAMILIES)
def test_state_validation_and_unfitted_config(factory, family):
    strategy = inflation_from_config(factory(family=family).to_config())
    assert strategy.to_config()["family"] == family
    with pytest.raises(RuntimeError, match="not been fitted"):
        strategy.factor(dates([0]), dates([1]))
    config = factory(family=family).fit(observations()).to_config()
    for change in (
        {"version": 99},
        {"upper": -1},
        {"coefficients": [float("nan")]},
        {"knots": [1]},
    ):
        damaged = json.loads(json.dumps(config))
        damaged["fitted_state"].update(change)
        with pytest.raises(ValueError, match="state"):
            inflation_from_config(damaged)


@pytest.mark.parametrize("factory", STRATEGIES)
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize(
    "field,value",
    [
        ("y", 0),
        ("y", -1),
        ("y", np.nan),
        ("y", np.inf),
        ("y", "invalid"),
        ("weight", -1),
        ("weight", 0),
        ("weight", np.nan),
        ("weight", np.inf),
    ],
)
def test_rejects_invalid_targets_and_weights(factory, family, field, value):
    data = observations()
    data = replace(data, **{field: pl.Series(np.full(len(data), value))})
    with pytest.raises(ValueError):
        factory(family=family).fit(data)


@pytest.mark.parametrize("factory", STRATEGIES)
def test_poisson_requires_positive_target_with_positive_weight(factory):
    data = observations()
    target = np.zeros(len(data))
    target[0] = 1
    weight = np.ones(len(data))
    weight[0] = 0
    with pytest.raises(ValueError, match="positive target on a positive-weight row"):
        factory(family="Poisson").fit(
            replace(data, y=pl.Series(target), weight=pl.Series(weight))
        )


@pytest.mark.parametrize("factory", STRATEGIES)
def test_legacy_learned_configs_default_to_gamma(factory):
    data = observations()
    default = factory().fit(data)
    explicit = factory(family="Gamma").fit(data)
    assert default.to_config() == explicit.to_config()
    legacy = json.loads(json.dumps(default.to_config()))
    legacy.pop("family")
    loaded = inflation_from_config(legacy)
    assert loaded.to_config()["family"] == "Gamma"
    np.testing.assert_array_equal(
        loaded.factor(dates([-1, 1]), dates([3, 5])),
        default.factor(dates([-1, 1]), dates([3, 5])),
    )
    legacy.pop("fitted_state")
    assert inflation_from_config(legacy).fit(data).to_config() == default.to_config()


@pytest.mark.parametrize("factory", STRATEGIES)
def test_validates_controls_dates_and_training_support(factory):
    data = observations()
    with pytest.raises(ValueError, match="missing"):
        factory(control_features=["absent"]).fit(data)
    with pytest.raises(TypeError, match="cat_features"):
        factory().fit(replace(data, cat_features=()))
    with pytest.raises(ValueError, match="infinity"):
        factory().fit(
            replace(data, X=data.X.with_columns(pl.lit(np.inf).alias("size")))
        )
    with pytest.raises(ValueError, match="distinct"):
        factory().fit(data.take(range(2)))
    strategy = factory().fit(data)
    with pytest.raises(ValueError, match="equal length"):
        strategy.factor(dates([0]), dates([1, 2]))
    with pytest.raises(ValueError, match="null"):
        strategy.factor(dates([0]), pl.Series([None], dtype=pl.Date))
    with pytest.raises(TypeError, match="Date or Datetime"):
        strategy.factor(dates([0]), pl.Series(["2020-01-01"]))
    assert SplineInflation(n_knots=2).fit(data).factor(dates([0]), dates([1]))[0] > 0
    with pytest.raises(ValueError, match="5 distinct"):
        SplineInflation().fit(data.take([0, 12, 24, 36]))


@pytest.mark.parametrize("factory", STRATEGIES)
@pytest.mark.parametrize(
    "options",
    [
        {"alpha": -1},
        {"alpha": np.nan},
        {"extrapolation": "invalid"},
        {"damping_half_life_years": 0},
        {"damping_half_life_years": np.inf},
        {"control_features": ["a", "a"]},
        {"family": "normal"},
    ],
)
def test_rejects_invalid_settings(factory, options):
    with pytest.raises(ValueError):
        factory(**options)


@pytest.mark.parametrize("value", [1, 2.5, True])
def test_rejects_invalid_knot_count(value):
    with pytest.raises(ValueError, match="n_knots"):
        SplineInflation(n_knots=value)


def test_timestamp_precision_timezone_and_extreme_factor_validation():
    strategy = LogLinearInflation(alpha=1e-8).fit(observations())
    a = pl.Series([datetime(2020, 1, 1, 12, tzinfo=timezone.utc)])
    b = pl.Series([datetime(2020, 1, 1, 18, tzinfo=timezone(timedelta(hours=6)))])
    np.testing.assert_array_equal(strategy.factor(a, b), [1.0])
    delta = pl.duration(hours=12)
    half_day = pl.select(pl.lit(a).dt.replace_time_zone(None) + delta).to_series()
    expected = np.exp(
        strategy.to_config()["fitted_state"]["coefficients"][0] / (365.25 * 2)
    )
    np.testing.assert_allclose(strategy.factor(a, half_day), [expected], rtol=1e-14)
    config = strategy.to_config()
    config["fitted_state"]["coefficients"] = [1000.0]
    huge = inflation_from_config(config)
    for start, end in [(0, 1), (1, 0)]:
        with pytest.raises(ValueError, match="finite and strictly positive"):
            huge.factor(dates([start]), dates([end]))


@pytest.mark.parametrize("factory", STRATEGIES)
@pytest.mark.parametrize("family", FAMILIES)
def test_adapter_training_isolated_from_validation_and_refits_state(factory, family):
    data = observations()
    train, validation = data.take(range(400)), data.take(range(400, len(data)))
    original = data.X.clone()
    first = InflationAdjustedModel(
        CatBoostRegressionModel(PARAMS), factory(family=family), ORIGIN.date()
    ).fit(train, validation)
    changed_validation = validation.with_target(validation.y * 100)
    second = InflationAdjustedModel(
        CatBoostRegressionModel(PARAMS), factory(family=family), ORIGIN.date()
    ).fit(train, changed_validation)
    assert first.inflation.to_config() == second.inflation.to_config()
    previous = first.inflation.to_config()
    first.fit(data)
    assert first.inflation.to_config() != previous
    assert second.inflation.to_config() == previous
    assert data.X.equals(original)


@pytest.mark.parametrize("factory", STRATEGIES)
@pytest.mark.parametrize("family", FAMILIES)
def test_adapter_changes_only_targets_and_allows_different_model_loss(
    factory, family, monkeypatch
):
    data = observations()
    original = (data.X.clone(), data.y.clone(), data.time.clone(), data.weight.clone())
    train, validation = data.take(range(400)), data.take(range(400, len(data)))
    captured = []
    original_fit = CatBoostRegressionModel.fit

    def record_fit(model, normalized_train, normalized_validation=None):
        captured.extend([normalized_train, normalized_validation])
        return original_fit(model, normalized_train, normalized_validation)

    monkeypatch.setattr(CatBoostRegressionModel, "fit", record_fit)
    loss = "Poisson" if family == "Gamma" else "Gamma"
    strategy = factory(family=family)
    adjusted = InflationAdjustedModel(
        CatBoostRegressionModel({**PARAMS, "loss_function": loss}),
        strategy,
        ORIGIN.date(),
    ).fit(train, validation)
    for raw, normalized in zip((train, validation), captured, strict=True):
        reference_dates = pl.Series([ORIGIN.date()] * len(raw))
        np.testing.assert_allclose(
            normalized.y.to_numpy(),
            raw.y.to_numpy() * strategy.factor(raw.time, reference_dates),
        )
        assert not normalized.y.equals(raw.y)
        assert normalized.weight.equals(raw.weight)
        assert normalized.X.equals(raw.X)
        assert normalized.time.equals(raw.time)
    for current, before in zip((data.X, data.y, data.time, data.weight), original, strict=True):
        assert current.equals(before)
    reference_prediction = adjusted.model.predict_pool(validation)
    expected = strategy.adjust(
        reference_prediction,
        pl.Series([ORIGIN.date()] * len(validation)),
        validation.time,
    )
    np.testing.assert_allclose(adjusted.predict_pool(validation), expected)


def test_fixed_strategies_validate_only_pool_dates_and_keep_legacy_configs():
    data = observations().with_target(np.zeros(588))
    fixed = ConstantRateInflation(0.03)
    assert fixed.fit(data) is fixed
    assert fixed.to_config() == {
        "strategy": "ConstantRateInflation",
        "annual_rate": 0.03,
    }
    monthly = ManualInflation({ORIGIN.date(): 1.0, datetime(2025, 1, 1).date(): 1.2})
    assert monthly.fit(data) is monthly
    for strategy in (fixed, monthly):
        restored = inflation_from_config(json.loads(json.dumps(strategy.to_config())))
        np.testing.assert_array_equal(
            strategy.factor(dates([0]), dates([1])),
            restored.factor(dates([0]), dates([1])),
        )
