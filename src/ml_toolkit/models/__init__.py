"""Model adapters used by training; project-defined factories also work."""

from .base import Model
from .catboost_classification import CatBoostModel as CatBoostClassificationModel
from .catboost_regression import CatBoostModel as CatBoostRegressionModel
from .inflation_regression import InflationAdjustedModel

__all__ = [
    "Model", "CatBoostClassificationModel", "CatBoostRegressionModel",
    "InflationAdjustedModel",
]
