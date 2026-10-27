"""Inflation adapter prediction dates and timestamp precision."""

import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml_toolkit.inflation import ConstantRateInflation, ManualInflation
from ml_toolkit.models import CatBoostRegressionModel, InflationAdjustedModel
from tests.models.support import PARAMS


def test_inflation_predicts_fragmented_frame_without_warning(data, monkeypatch):
    from dataclasses import replace

    data = replace(
        data,
        X=data.X.with_columns(
            [pl.col("signal").alias(f"feature_{i}") for i in range(105)]
        ),
    )
    model = InflationAdjustedModel(
        CatBoostRegressionModel(PARAMS), ConstantRateInflation(0.03), date(2024, 1, 1)
    ).fit(data)
    expected = model.predict_frame(
        data.X.with_columns(data.time.alias(model.prediction_date_column))
    )
    frame = data.to_model_input()
    frame.index = pd.Index(np.arange(len(frame))[::-1] + 100)
    original = frame.copy(deep=True)
    monkeypatch.setattr(type(data), "to_model_input", lambda self: frame)

    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        actual = model.predict_pool(data)

    np.testing.assert_allclose(actual, expected, rtol=1e-14)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("manual", [False, True])
def test_inflation_preserves_datetime_precision_and_validates_dates(data, manual):
    from dataclasses import replace
    from datetime import datetime

    timestamps = pl.Series(
        "date",
        [
            datetime(2024, 1, 1, 13, 30) + timedelta(days=i * 5)
            for i in range(len(data))
        ],
    )
    data = replace(data, time=timestamps)
    strategy = (
        ManualInflation({date(2024, 1, 1): 1.0, date(2025, 1, 1): 1.1})
        if manual
        else ConstantRateInflation(0.03)
    )
    model = InflationAdjustedModel(
        CatBoostRegressionModel(PARAMS), strategy, date(2024, 1, 1)
    ).fit(data)
    expected = strategy.adjust(
        model.model.predict_pool(data),
        pl.Series("date", [date(2024, 1, 1)] * len(data)),
        data.time,
    )
    np.testing.assert_allclose(model.predict_pool(data), expected, rtol=1e-14)
    frame = data.X.with_columns(data.time.alias(model.prediction_date_column))
    np.testing.assert_allclose(model.predict_frame(frame), expected, rtol=1e-14)
    with pytest.raises(ValueError, match="Missing model input column"):
        model.predict_frame(data.to_model_input())
    invalid = frame.to_pandas()
    invalid.loc[0, model.prediction_date_column] = pd.NaT
    with pytest.raises(ValueError, match="must not contain null"):
        model.predict_frame(invalid)
