from typing import Any

import numpy as np
from numpy.typing import NDArray

class Booster:
    params: dict[str, object]
    def __init__(
        self,
        params: dict[str, object] | None = ...,
        model_file: str | None = ...,
        **kwargs: Any,
    ) -> None: ...
    def feature_name(self) -> list[str]: ...
    def feature_importance(
        self, importance_type: str = ..., **kwargs: Any
    ) -> NDArray[np.float64]: ...
    def predict(self, data: object, **kwargs: Any) -> NDArray[np.float64]: ...
    def save_model(self, filename: str, **kwargs: Any) -> Booster: ...

class LGBMModel:
    booster_: Booster
