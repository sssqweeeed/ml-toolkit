"""Full training and serving model adapters."""
from .base import Model
from .catboost_regression import CatBoostRegressionModel, CatBoostLogLinkRegressionModel
from .catboost_classification import CatBoostClassificationModel, CatBoostPUClassificationModel
from .inflation_regression import InflationAdjustedModel

__all__ = [
    "Model", "CatBoostRegressionModel", "CatBoostLogLinkRegressionModel",
    "CatBoostClassificationModel", "CatBoostPUClassificationModel",
    "InflationAdjustedModel",
]
