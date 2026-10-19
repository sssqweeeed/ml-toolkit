from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray


def train_test_split(
    indices: NDArray[np.int64],
    *,
    test_size: float = ...,
    random_state: int | None = ...,
    stratify: NDArray[np.float64] | None = ...,
) -> list[NDArray[np.int64]]: ...

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
