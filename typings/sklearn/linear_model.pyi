from typing import Self

import numpy as np
from numpy.typing import NDArray


class GammaRegressor:
    coef_: NDArray[np.float64]

    def __init__(
        self, *, alpha: float = ..., max_iter: int = ..., tol: float = ...
    ) -> None: ...
    def fit(self, X: object, y: object, sample_weight: object = ...) -> Self: ...
