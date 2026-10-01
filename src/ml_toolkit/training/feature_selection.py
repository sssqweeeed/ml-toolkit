"""Optional model-driven feature elimination."""

from collections.abc import Callable
from typing import Any

import numpy as np

from .contracts import TrainingData


def feature_elimination(
    *, data: TrainingData, params: dict[str, Any], model_factory: Callable,
    num_features_to_select: int, validation_fraction: float = 0.25,
) -> TrainingData:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    split = int(len(data) * (1 - validation_fraction))
    if split <= 0 or split >= len(data):
        raise ValueError("Not enough rows for feature-selection validation")
    train = data.take(np.arange(split))
    validation = data.take(np.arange(split, len(data)))
    names = model_factory(params).select_features(
        train=train, validation=validation,
        num_features_to_select=num_features_to_select,
    )
    return data.select_features(names)
