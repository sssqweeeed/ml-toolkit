"""Array-returning built-in datasets used by the documentation gallery."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray


def load_digits(
    *,
    n_class: int = ...,
    return_X_y: Literal[True],
    as_frame: Literal[False] = ...,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]: ...
