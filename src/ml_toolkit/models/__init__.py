"""Model adapters for training and MLflow inference."""

from .base import Model, ShrinkableModel
from .catboost_classification import (
    CatBoostClassificationModel,
    CatBoostPUClassificationModel,
)
from .catboost_regression import CatBoostLogLinkRegressionModel, CatBoostRegressionModel
from .inflation_regression import InflationAdjustedModel

__all__ = [
    "Model",
    "ShrinkableModel",
    "CatBoostRegressionModel",
    "CatBoostLogLinkRegressionModel",
    "CatBoostClassificationModel",
    "CatBoostPUClassificationModel",
    "InflationAdjustedModel",
]
