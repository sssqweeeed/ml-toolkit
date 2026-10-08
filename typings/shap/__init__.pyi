from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

from . import plots as plots

class Explanation:
    values: NDArray[np.float64]
    base_values: NDArray[np.float64]
    feature_names: list[str] | None
    data: NDArray[Any]
    def __init__(
        self,
        values: object = ...,
        base_values: object = ...,
        data: object = ...,
        feature_names: list[str] | None = ...,
        **kwargs: Any,
    ) -> None: ...
    @property
    def abs(self) -> Self: ...
    def mean(self, axis: int) -> Self: ...
    def __getitem__(self, key: object) -> Self: ...

def summary_plot(values: object, features: object = ..., **kwargs: Any) -> None: ...
def dependence_plot(
    ind: object, shap_values: object, features: object, **kwargs: Any
) -> None: ...
