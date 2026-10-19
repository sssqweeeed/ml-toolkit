"""Inflation adapter prediction dates and timestamp precision."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml_toolkit.inflation import ConstantRateInflation, ManualInflation
from ml_toolkit.models import CatBoostRegressionModel, InflationAdjustedModel
from tests.models.support import PARAMS


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
