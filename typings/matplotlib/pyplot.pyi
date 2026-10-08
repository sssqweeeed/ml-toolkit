from typing import Any, Literal, overload

import numpy as np
from numpy.typing import NDArray

from .axes import Axes
from .figure import Figure

@overload
def subplots(
    nrows: Literal[1] = ..., ncols: Literal[1] = ..., **kwargs: Any
) -> tuple[Figure, Axes]: ...
@overload
def subplots(
    nrows: int = ..., ncols: int = ..., **kwargs: Any
) -> tuple[Figure, NDArray[np.object_]]: ...
def close(fig: Figure) -> None: ...

class Colormap:
    def __call__(self, value: float) -> tuple[float, float, float, float]: ...

def get_cmap(name: str) -> Colormap: ...
