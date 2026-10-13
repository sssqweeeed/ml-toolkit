"""Model adapters for training and MLflow inference."""

from .base import Model
from .catboost_classification import (
    CatBoostClassificationModel,
    CatBoostPUClassificationModel,
)
from .catboost_regression import CatBoostLogLinkRegressionModel, CatBoostRegressionModel
from .inflation_regression import InflationAdjustedModel

__all__ = [
    "Model",
    "CatBoostRegressionModel",
    "CatBoostLogLinkRegressionModel",
    "CatBoostClassificationModel",
    "CatBoostPUClassificationModel",
    "InflationAdjustedModel",
]
