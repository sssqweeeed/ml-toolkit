"""The benchmark uses chronological origins and complete calendar horizons."""

import numpy as np
import pytest

from examples.inflation_benchmark import benchmark, synthetic_log_index, synthetic_pool
from ml_toolkit.models import CatBoostLogLinkRegressionModel

pytestmark = pytest.mark.documentation


def test_benchmark_compares_identical_regressors_without_future_training(monkeypatch):
    data = synthetic_pool(months=18, rows_per_month=12)
    seen = []
    original = CatBoostLogLinkRegressionModel.fit

    def record(self, train, validation=None):
        assert validation is None
        seen.append((train.time.max(), self.params.copy()))
        return original(self, train, validation)

    monkeypatch.setattr(CatBoostLogLinkRegressionModel, "fit", record)
    result = benchmark(
        data,
        min_train_months=6,
        step_months=6,
        horizons=[1, 3, 12],
        iterations=3,
        known_log_index=synthetic_log_index(data.time),
    )
    assert result["method"].unique().len() == 4
    assert len(seen) == 8  # four strategies at each of two origins
    assert all(params == seen[0][1] for _, params in seen)
    assert result.filter(result["horizon_months"] == 12).height == 4
    assert (result["test_start"] >= result["origin"]).all()
    assert result["test_rows"].unique().to_list() == [12]
    assert all(
        end < origin
        for end, origin in zip(
            [seen[0][0], seen[4][0]], sorted(result["origin"].unique().to_list())
        )
    )
    for name in (
        "mae",
        "rmse",
        "gamma_deviance",
        "prediction_over_observed",
        "log_index_mae",
    ):
        assert np.isfinite(result[name].to_numpy()).all()


def test_benchmark_handles_unavailable_horizons_and_unknown_real_index():
    data = synthetic_pool(months=10, rows_per_month=8)
    result = benchmark(data, min_train_months=6, horizons=[1, 12], iterations=2)
    assert result["horizon_months"].unique().to_list() == [1]
    assert result["log_index_mae"].null_count() == len(result)
    with pytest.raises(ValueError, match="No complete"):
        benchmark(data, min_train_months=12)
    with pytest.raises(ValueError, match="positive integer"):
        benchmark(data, horizons=[0])
    with pytest.raises(ValueError, match="aligned"):
        benchmark(data, known_log_index=np.ones(2))
