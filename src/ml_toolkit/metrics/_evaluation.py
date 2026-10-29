"""Prediction validation and task scoring shared by training and diagnostics."""

import numpy as np

from ..data import Pool
from ..types import FloatArray
from .base import Metric


def validate_prediction(data: Pool, prediction: FloatArray) -> None:
    if prediction.shape != (len(data),) or not np.isfinite(prediction).all():
        raise ValueError("Predictions must contain one finite value per row")


def calculate_metrics(
    metric: Metric, data: Pool, prediction: FloatArray
) -> dict[str, float]:
    validate_prediction(data, prediction)
    return metric(
        data.y.to_numpy().astype(float),
        prediction,
        data.weight.to_numpy().astype(float),
    ).to_dict()
