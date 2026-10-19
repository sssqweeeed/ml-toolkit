"""Shared public types for numerical arrays and backend parameters.

FloatArray and IntArray are NumPy arrays with float64 and int64 values,
respectively. Individual APIs document their expected shapes. ModelParams
maps parameter names to backend-specific values.
"""

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
ModelParams: TypeAlias = dict[str, object]
