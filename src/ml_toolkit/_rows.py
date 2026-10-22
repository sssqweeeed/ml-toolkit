"""Positional row selection shared by pools and aligned predictions."""

from collections.abc import Sequence

import numpy as np

from .types import IntArray


def row_indexer(indices: Sequence[int] | IntArray, n_rows: int) -> slice | IntArray:
    """Use a view for contiguous in-bounds positions, otherwise preserve gather."""
    if (
        isinstance(indices, range)
        and indices.step == 1
        and 0 <= indices.start < indices.stop <= n_rows
    ):
        return slice(indices.start, indices.stop)
    index = np.asarray(indices, dtype=np.int64)
    if (
        index.ndim == 1
        and index.size
        and 0 <= index[0]
        and index[-1] < n_rows
        and index[-1] - index[0] == index.size - 1
        and np.all(index[1:] > index[:-1])
    ):
        return slice(int(index[0]), int(index[-1]) + 1)
    return index
