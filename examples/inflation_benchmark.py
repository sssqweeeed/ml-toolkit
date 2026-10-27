"""Compare monetary regressors on rolling calendar forecast origins.

Run ``python -m examples.inflation_benchmark`` for a reproducible synthetic
experiment with a known, changing index and shifting product composition.
Use ``--dataset retail --min-train-months 6 --step-months 1 --horizons 1 3``
after explicitly preparing Online Retail with ``examples.prepare_data``.
No source data is downloaded implicitly. Results do not establish a universal
winner; forecast horizons and strategy settings are configurable.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_tweedie_deviance,
)

from ml_toolkit.data import Pool
from ml_toolkit.inflation import (
    ConstantRateInflation,
    LogLinearInflation,
    SplineInflation,
)
from ml_toolkit.models import CatBoostLogLinkRegressionModel, InflationAdjustedModel
from ml_toolkit.types import FloatArray


def _month_number(value: date) -> int:
    return value.year * 12 + value.month - 1


def _month_start(number: int) -> date:
    year, month = divmod(number, 12)
    return date(year, month + 1, 1)


def synthetic_log_index(time: pl.Series) -> FloatArray:
    """Known log-index in the synthetic experiment; unrelated to real inflation."""
    days = (time.dt.date() - date(2015, 1, 1)).dt.total_days().to_numpy()
    years = days / 365.25
    return 0.08 * years + 0.015 * years**2


def synthetic_pool(
    *, months: int = 60, rows_per_month: int = 100, seed: int = 42
) -> Pool:
    """Generate positive Gamma prices and an increasing share of premium products."""
    if months < 2 or rows_per_month < 1:
        raise ValueError("Supply at least two months and one row per month")
    rng = np.random.default_rng(seed)
    month = np.repeat(np.arange(months), rows_per_month)
    time = pl.Series(
        "time", [_month_start(_month_number(date(2015, 1, 1)) + int(m)) for m in month]
    )
    size = rng.normal(size=len(month))
    premium = rng.random(len(month)) < 0.1 + 0.8 * month / (months - 1)
    mean = np.exp(4 + 0.3 * size + 1.2 * premium + synthetic_log_index(time))
    return Pool(
        X=pl.DataFrame(
            {"size": size, "category": np.where(premium, "premium", "budget")}
        ),
        y=pl.Series("price", rng.gamma(20, mean / 20)),
        time=time,
        weight=pl.Series("weight", rng.uniform(0.5, 1.5, len(month))),
        cat_features=("category",),
        numerical_features=("size",),
    )


def benchmark(
    data: Pool,
    *,
    horizons: Sequence[int] = (1, 3, 6, 12),
    min_train_months: int = 24,
    step_months: int = 6,
    iterations: int = 100,
    alpha: float = 0.001,
    n_knots: int = 5,
    extrapolation: str = "linear",
    damping_half_life_years: float = 1.0,
    constant_annual_rate: float = 0.08,
    known_log_index: FloatArray | None = None,
) -> pl.DataFrame:
    """Return fold metrics for a single held-out month at each requested horizon.

    A one-month horizon evaluates the first month after each training cutoff;
    a twelve-month horizon evaluates the twelfth month. Fit each model once per
    origin on preceding rows only, with identical CatBoost settings and no early
    stopping. Skip horizons beyond the available history, rather than report
    partial horizons. Each metric uses the held-out row weights. When supplied,
    known_log_index must align with data and enables log-index error measurement.
    """
    from ml_toolkit.data import validate_training_data

    validate_training_data(data)
    if not horizons or any(type(h) is not int or h < 1 for h in horizons):
        raise ValueError("horizons must contain positive integer months")
    if min_train_months < 1 or step_months < 1 or iterations < 1:
        raise ValueError("Training months, step months and iterations must be positive")
    if known_log_index is not None:
        known_log_index = np.asarray(known_log_index, dtype=float)
        if (
            known_log_index.shape != (len(data),)
            or not np.isfinite(known_log_index).all()
        ):
            raise ValueError("known_log_index must be finite and aligned with data")
    options = dict(
        alpha=alpha,
        extrapolation=extrapolation,
        damping_half_life_years=damping_half_life_years,
    )
    # Validate configurations even when the available history yields no folds.
    LogLinearInflation(**options)
    SplineInflation(n_knots=n_knots, **options)
    ConstantRateInflation(constant_annual_rate)
    calendar = data.time.dt.date()
    first = _month_number(calendar[0])
    last = _month_number(calendar[-1])
    rows: list[dict[str, object]] = []
    params = {
        "iterations": iterations,
        "depth": 4,
        "loss_function": "Gamma",
        "random_seed": 42,
        "thread_count": 2,
        "verbose": False,
        "allow_writing_files": False,
    }
    for origin in range(first + min_train_months, last + 1, step_months):
        eligible = [h for h in sorted(set(horizons)) if origin + h - 1 <= last]
        if not eligible:
            continue
        train_indices = np.flatnonzero((calendar < _month_start(origin)).to_numpy())
        if len(train_indices) == 0:
            continue
        train = data.take(train_indices)
        reference = train.time.dt.date()[-1]
        strategies = {
            "none": None,
            "constant": ConstantRateInflation(constant_annual_rate),
            "log_linear": LogLinearInflation(**options),
            "spline": SplineInflation(n_knots=n_knots, **options),
        }
        for name, strategy in strategies.items():
            regressor = CatBoostLogLinkRegressionModel(params)
            model = (
                regressor
                if strategy is None
                else InflationAdjustedModel(regressor, strategy, reference)
            )
            model.fit(train)
            for horizon in eligible:
                start = _month_start(origin + horizon - 1)
                end = _month_start(origin + horizon)
                indices = np.flatnonzero(
                    ((calendar >= start) & (calendar < end)).to_numpy()
                )
                if len(indices) == 0:
                    continue
                heldout = data.take(indices)
                prediction = model.predict_pool(heldout)
                target, weight = heldout.y.to_numpy(), heldout.weight.to_numpy()
                if weight.sum() <= 0:
                    continue
                index_error = None
                if known_log_index is not None:
                    estimated = (
                        np.zeros(len(heldout))
                        if strategy is None
                        else np.log(
                            strategy.factor(
                                pl.Series([reference] * len(heldout)), heldout.time
                            )
                        )
                    )
                    actual = (
                        known_log_index[indices] - known_log_index[train_indices[-1]]
                    )
                    index_error = float(
                        np.average(abs(estimated - actual), weights=weight)
                    )
                rows.append(
                    {
                        "method": name,
                        "origin": _month_start(origin),
                        "horizon_months": horizon,
                        "test_start": start,
                        "test_end": end,
                        "train_rows": len(train),
                        "test_rows": len(heldout),
                        "mae": float(
                            mean_absolute_error(
                                target, prediction, sample_weight=weight
                            )
                        ),
                        "rmse": float(
                            np.sqrt(
                                mean_squared_error(
                                    target, prediction, sample_weight=weight
                                )
                            )
                        ),
                        "gamma_deviance": float(
                            mean_tweedie_deviance(
                                target, prediction, power=2, sample_weight=weight
                            )
                        ),
                        "prediction_over_observed": float(
                            np.sum(weight * prediction) / np.sum(weight * target)
                        ),
                        "log_index_mae": index_error,
                    }
                )
    if not rows:
        raise ValueError(
            "No complete forecast months are available; reduce min_train_months or horizons"
        )
    return pl.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("synthetic", "retail"), default="synthetic"
    )
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 6, 12])
    parser.add_argument("--min-train-months", type=int, default=24)
    parser.add_argument("--step-months", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--n-knots", type=int, default=5)
    parser.add_argument(
        "--extrapolation",
        choices=("linear", "constant", "error", "damped"),
        default="linear",
    )
    parser.add_argument("--damping-half-life-years", type=float, default=1.0)
    parser.add_argument("--constant-annual-rate", type=float, default=0.08)
    parser.add_argument(
        "--output", type=Path, help="Optional CSV file for individual fold results"
    )
    args = parser.parse_args()
    if args.dataset == "synthetic":
        data = synthetic_pool()
        truth = synthetic_log_index(data.time)
    else:
        from examples.retail_data import load_retail_pool

        data = load_retail_pool(download_if_missing=False)
        truth = None
    result = benchmark(
        data,
        horizons=args.horizons,
        min_train_months=args.min_train_months,
        step_months=args.step_months,
        iterations=args.iterations,
        alpha=args.alpha,
        n_knots=args.n_knots,
        extrapolation=args.extrapolation,
        damping_half_life_years=args.damping_half_life_years,
        constant_annual_rate=args.constant_annual_rate,
        known_log_index=truth,
    )
    if args.output is not None:
        result.write_csv(args.output)
    metrics = ["mae", "rmse", "gamma_deviance", "prediction_over_observed"]
    if truth is not None:
        metrics.append("log_index_mae")
    summary = (
        result.group_by("method", "horizon_months")
        .agg(pl.len().alias("folds"), *[pl.col(name).mean() for name in metrics])
        .sort("horizon_months", "method")
    )
    with pl.Config(tbl_rows=100, tbl_cols=12, tbl_width_chars=160):
        print(summary)
    print("Metrics are equally averaged across origins; each fold uses row weights.")
    print(
        "The constant rate is an assumption. Results depend on this dataset and configuration."
    )


if __name__ == "__main__":
    main()
