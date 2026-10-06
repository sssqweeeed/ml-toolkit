"""Optional selection implemented by each model adapter."""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from ..config import FeatureSelectionConfig
from ..models.base import Model
from .contracts import TrainingData, validate_training_data

logger = logging.getLogger(__name__)


def feature_elimination(
    *,
    data: TrainingData,
    params: dict[str, Any],
    model_factory: Callable[[dict[str, Any]], Model],
    config: FeatureSelectionConfig,
) -> TrainingData:
    """Select features on a chronological internal holdout and return all rows.

    Parameters
    ----------
    data : TrainingData
        Chronological selection data, excluding outer CV validation rows.
    params : dict[str, Any]
        Native parameters for the temporary model; copied before factory use.
    model_factory : Callable[[dict[str, Any]], Model]
        Factory whose adapter implements select_features.
    config : FeatureSelectionConfig
        Retained feature count, internal holdout fraction, and backend options.

    Returns
    -------
    TrainingData
        All original selection rows restricted to the selected feature names.

    Raises
    ------
    ValueError
        Data are invalid, too few rows remain in either split, or n_features
        exceeds the input feature count.
    """
    validate_training_data(data)
    if config.n_features > len(data.X.columns):
        raise ValueError("n_features must not exceed the input feature count")
    split = int(len(data) * (1 - config.validation_fraction))
    if split <= 0 or split >= len(data):
        raise ValueError("Not enough rows for feature-selection validation")
    train = data.take(np.arange(split))
    validation = data.take(np.arange(split, len(data)))
    logger.info("Starting feature selection: %s features requested", config.n_features)
    logger.debug(
        "Feature selection: %s training rows, %s validation rows",
        len(train),
        len(validation),
    )
    names = model_factory(params.copy()).select_features(
        train=train, validation=validation, config=config
    )
    selected = data.select_features(names)
    logger.info("Completed feature selection: %s features selected", len(names))
    logger.debug("Selected features: %s", names)
    return selected
