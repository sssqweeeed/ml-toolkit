import numpy as np
from numpy.typing import NDArray


class BSpline:
    t: NDArray[np.float64]

    def __init__(
        self, t: object, c: object, k: int, *, extrapolate: bool = ...
    ) -> None: ...
    def __call__(self, x: object, nu: int = ...) -> NDArray[np.float64]: ...
