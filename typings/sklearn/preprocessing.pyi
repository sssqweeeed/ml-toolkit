import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline
from scipy.sparse import csr_matrix


class SplineTransformer:
    bsplines_: list[BSpline]

    def __init__(
        self,
        *,
        n_knots: int = ...,
        degree: int = ...,
        knots: str = ...,
        include_bias: bool = ...,
    ) -> None: ...
    def fit_transform(self, X: object) -> NDArray[np.float64]: ...


class StandardScaler:
    def fit_transform(
        self, X: object, *, sample_weight: object = ...
    ) -> NDArray[np.float64]: ...


class OneHotEncoder:
    def __init__(self, *, sparse_output: bool = ...) -> None: ...
    def fit_transform(self, X: object) -> csr_matrix: ...
