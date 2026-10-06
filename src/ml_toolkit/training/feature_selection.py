"""One top-k importance selection shared by every training adapter."""

import logging

import catboost
import numpy as np
from sklearn.feature_selection import SelectFromModel

from .contracts import validate_training_data

logger = logging.getLogger(__name__)


def _catboost_importance(estimator):
    return estimator.get_feature_importance(type="PredictionValuesChange")


def _select_features(*, data, params, model_factory, config):
    validate_training_data(data)
    names = data.X.columns
    if config.n_features > len(names):
        raise ValueError("n_features must not exceed the input feature count")
    if config.n_features == len(names):
        return data
    split = int(len(data) * (1 - config.validation_fraction))
    if split <= 0 or split >= len(data):
        raise ValueError("Not enough rows for feature-selection validation")
    train = data.take(np.arange(split))
    validation = data.take(np.arange(split, len(data)))
    logger.info("Starting feature selection: %s features requested", config.n_features)
    model = model_factory(params.copy()).fit(train, validation)
    native = model.native_model
    if isinstance(native, catboost.CatBoost):
        getter = _catboost_importance
        native_names = list(native.feature_names_)
    else:
        getter = "auto"
        native_names = list(getattr(native, "feature_names_in_", names))
    if len(native_names) != len(names) or set(native_names) != set(names):
        raise ValueError(
            "Feature selection requires native features matching the input columns"
        )
    selector = SelectFromModel(
        native,
        prefit=True,
        threshold=-np.inf,
        max_features=config.n_features,
        importance_getter=getter,
    )
    try:
        mask = selector.get_support()
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            "Feature selection requires a fitted native model exposing "
            "feature importances or coefficients"
        ) from error
    if mask.shape != (len(names),):
        raise ValueError(
            "Native feature importance dimensions do not match the input columns"
        )
    selected = {name for name, keep in zip(native_names, mask, strict=True) if keep}
    ordered = [name for name in names if name in selected]
    logger.debug("Selected features: %s", ordered)
    logger.info("Completed feature selection: %s features selected", len(ordered))
    return data.select_features(ordered)
