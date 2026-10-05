"""Optional model-driven feature elimination."""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from .contracts import TrainingData

logger = logging.getLogger(__name__)


def feature_elimination(
    *, data: TrainingData, params: dict[str, Any], model_factory: Callable,
    num_features_to_select: int, validation_fraction: float = 0.25,
) -> TrainingData:
    """Select features using a chronological holdout and return reduced data.

    Parameters
    ----------
    data : TrainingData
        Project-owned data in chronological order. All rows are returned with
        the selected columns.
    params : dict[str, Any]
        Model parameters used for feature selection.
    model_factory : Callable
        Factory receiving a parameter dictionary and returning a model
        implementing the Model protocol.
    num_features_to_select : int
        Number of features requested from the model selector.
    validation_fraction : float, default 0.25
        Fraction of trailing rows used as validation; strictly between zero and
        one.

    Returns
    -------
    TrainingData
        Data returned by data.select_features with the model-selected column
        names.

    Raises
    ------
    ValueError
        The validation fraction is invalid or the split leaves an empty
        partition.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    split = int(len(data) * (1 - validation_fraction))
    if split <= 0 or split >= len(data):
        raise ValueError("Not enough rows for feature-selection validation")
    train = data.take(np.arange(split))
    validation = data.take(np.arange(split, len(data)))
    logger.info("Starting feature selection: %s features requested", num_features_to_select)
    logger.debug(
        "Feature selection: %s training rows, %s validation rows",
        len(train),
        len(validation),
    )
    names = model_factory(params).select_features(
        train=train, validation=validation,
        num_features_to_select=num_features_to_select,
    )
    selected = data.select_features(names)
    logger.info("Completed feature selection: %s features selected", len(names))
    logger.debug("Selected features: %s", names)
    return selected
