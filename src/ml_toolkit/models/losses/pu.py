"""Training-only PU objectives, risk metrics and the nnPU branch correction.

Numba-compiled derivatives and risk components, CatBoost objective and metric
classes, and the state shared by the nnPU objective and its learn-set metric.
The whole training pool is the unlabeled sample of the population; labeled
rows are also part of it. Serving never imports this module.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Self, TypeAlias, cast

import numpy as np
from numba import njit
from numpy.typing import NDArray

from ...metrics.pu import risk_components
from ...types import FloatArray

# CatBoost supplies float32 targets/weights and float64 CPU approximations.
PoolArray: TypeAlias = NDArray[np.float32] | FloatArray


def _derivatives(
    logits: FloatArray,
    labels: PoolArray,
    weights: PoolArray | None,
    positive_scale: float,
    reverse: bool,
    gamma: float,
) -> FloatArray:
    # CatBoost CPU convention: der1 = -dL/dz, der2 = -d2L/dz2, times row weight.
    result = np.empty((len(labels), 2), dtype=np.float64)
    for index in range(len(labels)):
        weight = 1.0 if weights is None else float(weights[index])
        logit = logits[index]
        if logit >= 0:
            probability = 1.0 / (1.0 + math.exp(-logit))
        else:
            exp_logit = math.exp(logit)
            probability = exp_logit / (1.0 + exp_logit)
        curvature = probability * (1.0 - probability)
        positive = labels[index] == 1
        if reverse:
            # Minimize -negative_risk, scaled by gamma.
            factor = gamma * (1.0 - positive_scale) if positive else gamma
            der1 = factor * probability
            der2 = factor * curvature
        else:
            der1 = positive_scale - probability if positive else -probability
            der2 = -curvature
        result[index, 0] = weight * der1
        result[index, 1] = weight * der2
    return result


def _risk_components(
    logits: FloatArray,
    labels: PoolArray,
    weights: PoolArray | None,
    prior: float,
) -> tuple[float, float]:
    positive_softplus = 0.0
    all_softplus = 0.0
    positive_weight = 0.0
    total_weight = 0.0
    positive_loss = 0.0
    # Same estimator as metrics.pu.risk_components (a test ties them); compiled
    # because CatBoost evaluates it on the full learn set after every tree.
    for index in range(len(labels)):
        logit = logits[index]
        if not math.isfinite(logit):
            raise ValueError("PU risk requires finite raw predictions")
        weight = 1.0 if weights is None else float(weights[index])
        tail = math.log1p(math.exp(-abs(logit)))
        softplus = max(logit, 0.0) + tail
        all_softplus += weight * softplus
        total_weight += weight
        if labels[index] == 1:
            positive_softplus += weight * softplus
            positive_weight += weight
            positive_loss += weight * (max(-logit, 0.0) + tail)
    if positive_weight <= 0.0 or total_weight <= 0.0:
        raise ValueError("PU risk needs positive weight in both groups")
    positive_risk = prior * positive_loss / positive_weight
    negative_risk = (
        all_softplus / total_weight - prior * positive_softplus / positive_weight
    )
    return positive_risk, negative_risk


# Numba's untyped dispatcher is confined to this third-party boundary. The
# compiled functions retain the concrete array contracts used by the adapter.
pu_derivatives = cast(
    Callable[
        [FloatArray, PoolArray, PoolArray | None, float, bool, float], FloatArray
    ],
    njit(cache=True, nogil=True)(_derivatives),
)
pu_risk_components = cast(
    Callable[[FloatArray, PoolArray, PoolArray | None, float], tuple[float, float]],
    njit(cache=True, nogil=True)(_risk_components),
)


def pu_risks(
    logits: FloatArray,
    labels: FloatArray,
    weights: FloatArray,
    class_prior: float,
) -> tuple[float, float]:
    """Return the positive and estimated negative logistic PU risks."""
    logits = np.asarray(logits, dtype=float).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if not (logits.shape == labels.shape == weights.shape):
        raise ValueError("PU logits, labels, and weights must have equal lengths")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("PU labels must be 1 (labeled positive) or 0 (unlabeled)")
    if not np.isfinite(logits).all() or not np.isfinite(weights).all():
        raise ValueError("PU logits and weights must be finite")
    if (weights < 0).any():
        raise ValueError("PU weights must be non-negative")
    if not 0 < class_prior < 1:
        raise ValueError("class_prior must be strictly between 0 and 1")
    positive = labels == 1
    unlabeled = ~positive
    if weights[positive].sum() <= 0 or weights[unlabeled].sum() <= 0:
        raise ValueError("PU training needs positive-weight rows in both groups")

    return risk_components(
        np.logaddexp(0.0, -logits),
        np.logaddexp(0.0, logits),
        labels,
        weights,
        class_prior,
    )


class NNPUCorrection:
    """Shared nnPU branch state between the objective and the learn metric.

    CatBoost evaluates the eval metric on the complete learn set after every
    tree and before the next tree's derivatives; that evaluation decides the
    next tree's branch (Kiryo et al. 2017, Algorithm 1, full-batch).
    CatBoost deep-copies loss and metric objects, so copies return self.
    """

    def __init__(self, beta: float, gamma: float, n_train: int) -> None:
        self.beta = beta
        self.gamma = gamma
        self.n_train = n_train
        # At the constant baseline the negative risk is (1 - prior) * softplus(b)
        # > 0, so the first tree always descends.
        self.reverse = False
        # Set by the objective, cleared by the learn-set metric call.
        self.pending_learn = False
        self.learn_evaluations = 0
        # Updates that selected the reversed branch; the last update after the
        # final tree can over-count by one.
        self.reversed_trees = 0

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        return self

    def __copy__(self) -> Self:
        return self

    def update(self, negative_risk: float) -> None:
        self.reverse = negative_risk < -self.beta
        self.pending_learn = False
        self.learn_evaluations += 1
        if self.reverse:
            self.reversed_trees += 1


class UPUObjective:
    # Python callbacks: skip CatBoost's numba compile attempt on every fit.
    no_jit = True

    def __init__(self, positive_scale: float) -> None:
        self.positive_scale = positive_scale

    def calc_ders_range(
        self, approxes: FloatArray, targets: PoolArray, weights: PoolArray | None
    ) -> FloatArray:
        return pu_derivatives(
            approxes, targets, weights, self.positive_scale, False, 1.0
        )


class NNPUObjective:
    no_jit = True

    def __init__(self, positive_scale: float, correction: NNPUCorrection) -> None:
        self.positive_scale = positive_scale
        self.correction = correction

    def calc_ders_range(
        self, approxes: FloatArray, targets: PoolArray, weights: PoolArray | None
    ) -> FloatArray:
        self.correction.pending_learn = True
        return pu_derivatives(
            approxes,
            targets,
            weights,
            self.positive_scale,
            self.correction.reverse,
            self.correction.gamma,
        )


class PURiskMetric:
    # Python callbacks: skip CatBoost's numba compile attempt on every fit.
    no_jit = True

    def __init__(self, class_prior: float, *, non_negative: bool) -> None:
        self.class_prior = class_prior
        self.non_negative = non_negative

    def is_max_optimal(self) -> bool:
        return False

    def get_final_error(self, error: float, weight: float) -> float:
        return error / weight

    def _components(
        self,
        approxes: Sequence[Sequence[float]],
        target: Sequence[float],
        weight: Sequence[float] | None,
    ) -> tuple[float, float]:
        if len(approxes) != 1:
            raise ValueError("PU risk requires one-dimensional raw predictions")
        # CatBoost calls this once per dataset with all rows.
        return pu_risk_components(
            np.asarray(approxes[0], dtype=np.float64),
            np.asarray(target),
            None if weight is None else np.asarray(weight),
            self.class_prior,
        )

    def evaluate(
        self,
        approxes: Sequence[Sequence[float]],
        target: Sequence[float],
        weight: Sequence[float] | None,
    ) -> tuple[float, float]:
        positive, negative = self._components(approxes, target, weight)
        return positive + (max(0.0, negative) if self.non_negative else negative), 1.0


class UPUMetric(PURiskMetric):
    def __init__(self, class_prior: float) -> None:
        super().__init__(class_prior, non_negative=False)


class NNPUMetric(PURiskMetric):
    def __init__(
        self, class_prior: float, correction: NNPUCorrection | None = None
    ) -> None:
        super().__init__(class_prior, non_negative=True)
        self.correction = correction

    def evaluate(
        self,
        approxes: Sequence[Sequence[float]],
        target: Sequence[float],
        weight: Sequence[float] | None,
    ) -> tuple[float, float]:
        positive, negative = self._components(approxes, target, weight)
        correction = self.correction
        # The learn set is evaluated right after the derivatives and before the
        # eval set; the size check is extra insurance.
        if (
            correction is not None
            and correction.pending_learn
            and len(target) == correction.n_train
        ):
            correction.update(negative)
        return positive + max(0.0, negative), 1.0
