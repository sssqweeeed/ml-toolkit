"""Feature selection across native, PU, and inflation-adjusted adapters."""

from datetime import date

import numpy as np
import polars as pl
import pytest

from ml_toolkit.config import HoldoutConfig
from ml_toolkit.feature_selection import RFESelector
from ml_toolkit.inflation import ConstantRateInflation
from ml_toolkit.models import (
    CatBoostClassificationModel,
    CatBoostLogLinkRegressionModel,
    CatBoostPUClassificationModel,
    CatBoostRegressionModel,
    InflationAdjustedModel,
)
from ml_toolkit.training import ShapRanker, importance_ranker, split_holdout
from tests.models.support import PARAMS
from tests.support.training import mae


@pytest.mark.parametrize(
    "kind", ["regression", "log_link", "classification", "uPU", "nnPU", "inflation"]
)
@pytest.mark.parametrize(
    "ranker", [importance_ranker, ShapRanker(max_samples=5, seed=42)]
)
def test_rfe_selection_preserves_adapter_policies_and_original_order(
    data, kind, ranker
):
    if kind in ("classification", "uPU", "nnPU"):
        data = data.with_target(pl.Series("target", [1, 0, 0] * 16))
    fitted = []

    def factory(params):
        if kind == "regression":
            model = CatBoostRegressionModel(params)
        elif kind == "log_link":
            model = CatBoostLogLinkRegressionModel(params)
        elif kind == "classification":
            model = CatBoostClassificationModel(params)
        elif kind in ("uPU", "nnPU"):
            model = CatBoostPUClassificationModel(
                {**params, "loss": kind, "pu_prior": 0.2}
            )
        else:
            model = InflationAdjustedModel(
                CatBoostLogLinkRegressionModel(params),
                ConstantRateInflation(0.03),
                date(2024, 1, 1),
            )
        fitted.append(model)
        return model

    from ml_toolkit.metrics import PUMetric

    metric = PUMetric(0.2, loss=kind) if kind in ("uPU", "nnPU") else mae
    metric_name = "pu_risk" if kind in ("uPU", "nnPU") else "mae"
    partition = split_holdout(data.take(range(32)), HoldoutConfig(val_size=0.25))
    result = RFESelector(
        ranker=ranker, min_features=2, metric=metric_name, direction="minimize"
    ).select(
        metric=metric,
        train=partition.train,
        validation=partition.val,
        params=PARAMS,
        model_factory=factory,
    )
    assert len(fitted) == len(result.steps) == 2
    assert [len(step.features) for step in result.steps] == [3, 2]
    assert result.features == result.steps[result.selected_step].features
    assert list(result.features) == [
        name for name in data.X.columns if name in result.features
    ]
    assert all(np.isfinite(step.metrics[metric_name]) for step in result.steps)


def test_inflation_selection_fits_strategy_on_supplied_training_only(data):
    seen = []

    class RecordingInflation(ConstantRateInflation):
        def fit(self, dates, y, sample_weight=None):
            seen.append((dates.clone(), y.copy(), sample_weight.copy()))
            return super().fit(dates, y, sample_weight)

    partition = split_holdout(data.take(range(32)), HoldoutConfig(val_size=0.25))
    RFESelector(
        ranker=importance_ranker, min_features=2, metric="mae", direction="minimize"
    ).select(
        metric=mae,
        train=partition.train,
        validation=partition.val,
        params={},
        model_factory=lambda params: InflationAdjustedModel(
            CatBoostRegressionModel(PARAMS), RecordingInflation(0.03), date(2024, 1, 1)
        ),
    )
    assert len(seen) == 2
    for dates, target, weights in seen:
        assert dates.to_list() == partition.train.time.to_list()
        np.testing.assert_array_equal(target, partition.train.y.to_numpy())
        np.testing.assert_array_equal(weights, partition.train.weight.to_numpy())
