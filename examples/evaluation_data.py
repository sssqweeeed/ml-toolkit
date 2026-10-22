"""Fit chronological Bike Sharing examples used by the evaluation gallery."""

from dataclasses import dataclass

import numpy as np
import polars as pl

from examples.gallery_data import load_bike_pool
from ml_toolkit.data import Pool
from ml_toolkit.models import CatBoostClassificationModel, CatBoostRegressionModel
from ml_toolkit.training import Holdout
from ml_toolkit.types import FloatArray, ModelParams


@dataclass(frozen=True)
class BikeEvaluation:
    """Keep the fitted model and aligned train/test report inputs together."""

    data: Pool
    train: Pool
    test: Pool
    model: CatBoostClassificationModel | CatBoostRegressionModel
    prediction: FloatArray
    split: pl.Series


def fit_bike_evaluation(*, classification: bool = False) -> BikeEvaluation:
    """Fit 2011 hours and predict all recorded hours, retaining 2012 as test.

    The optional binary target means at least 200 rentals in a recorded hour.
    This fixed threshold defines the task before splitting. Source timestamps,
    feature values and unit hourly weights are unchanged. Requires the explicit
    cache preparation described in the documentation; never downloads data.
    """
    data = load_bike_pool(download_if_missing=False)
    if classification:
        data = data.with_target((data.y >= 200).cast(pl.Float64))
    final = Holdout(data, test_size="2012-01-01").get_final()
    train, test = final.train, final.heldout
    assert test is not None
    params: ModelParams = {
        "iterations": 80,
        "depth": 5,
        "learning_rate": 0.1,
        "random_seed": 42,
        "thread_count": 2,
        "verbose": False,
        "allow_writing_files": False,
    }
    model = (
        CatBoostClassificationModel({**params, "loss_function": "Logloss"})
        if classification
        else CatBoostRegressionModel({**params, "loss_function": "RMSE"})
    )
    model.fit(train)
    prediction = np.concatenate((model.predict_pool(train), model.predict_pool(test)))
    split = pl.Series("split", ["train"] * len(train) + ["test"] * len(test))
    return BikeEvaluation(data, train, test, model, prediction, split)
