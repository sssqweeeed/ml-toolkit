from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

class TimeSeriesSplit:
    def __init__(
        self,
        n_splits: int = ...,
        *,
        max_train_size: int | None = ...,
        test_size: int | None = ...,
        gap: int = ...,
    ) -> None: ...
    def split(
        self, X: object, y: object = ..., groups: object = ...
    ) -> Iterator[tuple[NDArray[np.int64], NDArray[np.int64]]]: ...
