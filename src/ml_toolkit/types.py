"""Shared array and parameter types used by toolkit APIs."""

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
ModelParams: TypeAlias = dict[str, object]
