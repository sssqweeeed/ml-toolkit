"""Numba-compiled Gamma derivatives and deviance for log-mean predictions."""

import math
from collections.abc import Callable, Sequence
from typing import TypeAlias, cast

import numpy as np
from numba import njit
from numpy.typing import NDArray

from ....types import FloatArray

PoolArray: TypeAlias = NDArray[np.float32] | FloatArray


def _derivatives(
    approxes: FloatArray, targets: PoolArray, weights: PoolArray | None
) -> FloatArray:
    result = np.zeros((len(targets), 2), dtype=np.float64)
    for index in range(len(targets)):
        weight = 1.0 if weights is None else float(weights[index])
        if weight == 0.0:
            continue
        # Unlike float(), np.float64 explicitly promotes Numba's float32 labels
        # before log; otherwise precision is lost even with float64 raw scores.
        z = math.log(np.float64(targets[index])) - approxes[index]
        der1 = weight * math.expm1(z)
        der2 = -weight * math.exp(z)
        if not math.isfinite(approxes[index]) or not (
            math.isfinite(der1) and math.isfinite(der2)
        ):
            raise ValueError(
                "Gamma derivatives require finite raw predictions and results"
            )
        result[index, 0], result[index, 1] = der1, der2
    return result


def _deviance(
    approxes: FloatArray, targets: PoolArray, weights: PoolArray | None
) -> tuple[float, float]:
    error, total_weight = 0.0, 0.0
    for index in range(len(targets)):
        weight = 1.0 if weights is None else float(weights[index])
        if weight == 0.0:
            continue
        z = math.log(np.float64(targets[index])) - approxes[index]
        error += 2.0 * weight * (math.expm1(z) - z)
        total_weight += weight
    if not math.isfinite(error) or not math.isfinite(total_weight):
        raise ValueError("Gamma deviance requires finite raw predictions and results")
    return error, total_weight


gamma_derivatives = cast(
    Callable[[FloatArray, PoolArray, PoolArray | None], FloatArray],
    njit(cache=True, nogil=True)(_derivatives),
)
gamma_deviance = cast(
    Callable[[FloatArray, PoolArray, PoolArray | None], tuple[float, float]],
    njit(cache=True, nogil=True)(_deviance),
)


class GammaObjective:
    no_jit = True

    def calc_ders_range(
        self, approxes: FloatArray, targets: PoolArray, weights: PoolArray | None
    ) -> FloatArray:
        return gamma_derivatives(approxes, targets, weights)


class GammaMetric:
    no_jit = True

    def is_max_optimal(self) -> bool:
        return False

    def get_final_error(self, error: float, weight: float) -> float:
        if weight <= 0 or not math.isfinite(weight) or not math.isfinite(error):
            raise ValueError("Gamma deviance requires finite error and positive weight")
        return error / weight

    def evaluate(
        self,
        approxes: Sequence[Sequence[float]],
        target: Sequence[float],
        weight: Sequence[float] | None,
    ) -> tuple[float, float]:
        if len(approxes) != 1:
            raise ValueError("Gamma deviance requires one-dimensional raw predictions")
        return gamma_deviance(
            np.asarray(approxes[0], dtype=np.float64),
            np.asarray(target),
            None if weight is None else np.asarray(weight),
        )
