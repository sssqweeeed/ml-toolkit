"""Compiled CPU computations for CatBoost's unbiased PU objective."""

import math
from collections.abc import Callable
from typing import TypeAlias, cast

import numpy as np
from numba import njit
from numpy.typing import NDArray

from ....types import FloatArray

# CatBoost supplies float32 targets/weights and float64 CPU approximations.
PoolArray: TypeAlias = NDArray[np.float32] | FloatArray


def _derivatives(
    logits: FloatArray,
    labels: PoolArray,
    weights: PoolArray | None,
    positive_scale: float,
    unlabeled_scale: float,
) -> FloatArray:
    result = np.empty((len(labels), 2), dtype=np.float64)
    for index in range(len(labels)):
        weight = 1.0 if weights is None else float(weights[index])
        if labels[index] == 1:
            result[index, 0] = weight * positive_scale
            result[index, 1] = 0.0
        else:
            logit = logits[index]
            if logit >= 0:
                probability = 1.0 / (1.0 + math.exp(-logit))
            else:
                exp_logit = math.exp(logit)
                probability = exp_logit / (1.0 + exp_logit)
            result[index, 0] = -weight * unlabeled_scale * probability
            result[index, 1] = (
                -weight * unlabeled_scale * probability * (1.0 - probability)
            )
    return result


def _risk(
    logits: FloatArray,
    labels: PoolArray,
    weights: PoolArray | None,
    prior: float,
) -> float:
    positive_logit = 0.0
    unlabeled_loss = 0.0
    positive_weight = 0.0
    unlabeled_weight = 0.0
    for index in range(len(labels)):
        logit = logits[index]
        if not math.isfinite(logit):
            raise ValueError("PU risk requires finite raw predictions")
        weight = 1.0 if weights is None else float(weights[index])
        if labels[index] == 1:
            positive_logit += weight * logit
            positive_weight += weight
        else:
            unlabeled_loss += weight * (
                max(logit, 0.0) + math.log1p(math.exp(-abs(logit)))
            )
            unlabeled_weight += weight
    if positive_weight <= 0.0 or unlabeled_weight <= 0.0:
        raise ValueError("PU risk needs positive weight in both groups")
    # softplus(-z) - softplus(z) = -z: avoid subtracting two large P losses.
    return unlabeled_loss / unlabeled_weight - prior * positive_logit / positive_weight


# Numba's untyped dispatcher is confined to this third-party boundary. The
# compiled functions retain the concrete array contracts used by the adapter.
upu_derivatives = cast(
    Callable[[FloatArray, PoolArray, PoolArray | None, float, float], FloatArray],
    njit(cache=True, nogil=True)(_derivatives),
)
upu_risk = cast(
    Callable[[FloatArray, PoolArray, PoolArray | None, float], float],
    njit(cache=True, nogil=True)(_risk),
)


class UPUObjective:
    no_jit = True

    def __init__(self, positive_scale: float, unlabeled_scale: float) -> None:
        self.positive_scale = positive_scale
        self.unlabeled_scale = unlabeled_scale

    def calc_ders_range(
        self, approxes: FloatArray, targets: PoolArray, weights: PoolArray | None
    ) -> FloatArray:
        return upu_derivatives(
            approxes, targets, weights, self.positive_scale, self.unlabeled_scale
        )
