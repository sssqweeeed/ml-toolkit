from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, Self

import numpy as np

if TYPE_CHECKING:
    from ..training.contracts import TrainingData


class Model(Protocol):
    @property
    def native_model(self) -> Any: ...

    @property
    def best_iteration(self) -> int | None: ...

    def fit(self, train: TrainingData, validation: TrainingData | None = None) -> Self: ...

    def predict_pool(self, data: TrainingData) -> np.ndarray: ...

    def select_features(
        self,
        train: TrainingData,
        validation: TrainingData,
        num_features_to_select: int,
    ) -> list[str]: ...

    def log(self, sample: TrainingData) -> Any: ...
