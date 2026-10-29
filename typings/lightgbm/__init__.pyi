from typing import Any

import numpy as np
from numpy.typing import NDArray

class Booster:
    params: dict[str, object]
    best_iteration: int
    def __init__(
        self,
        params: dict[str, object] | None = ...,
        model_file: str | None = ...,
        **kwargs: Any,
    ) -> None: ...
    def feature_name(self) -> list[str]: ...
    def current_iteration(self) -> int: ...
    def model_to_string(
        self, num_iteration: int | None = ..., start_iteration: int = ...,
        importance_type: str = ...,
    ) -> str: ...
    def feature_importance(
        self, importance_type: str = ..., **kwargs: Any
    ) -> NDArray[np.float64]: ...
    def predict(self, data: object, **kwargs: Any) -> NDArray[np.float64]: ...
    def save_model(self, filename: str, **kwargs: Any) -> Booster: ...

class LGBMModel:
    booster_: Booster
