"""PU risk metrics and the nnPU objective, without eager compiler imports."""

import math
from collections.abc import Sequence

import numpy as np

from ....types import FloatArray


def pu_risks(
    logits: FloatArray,
    labels: FloatArray,
    weights: FloatArray,
    prior: float,
) -> tuple[float, float]:
    """Return the positive and estimated negative logistic PU risks."""
    from ....metrics._pu import risk_components

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
    if not 0 < prior < 1:
        raise ValueError("pu_prior must be strictly between 0 and 1")
    positive = labels == 1
    unlabeled = ~positive
    if weights[positive].sum() <= 0 or weights[unlabeled].sum() <= 0:
        raise ValueError("PU training needs positive-weight rows in both groups")

    return risk_components(
        np.logaddexp(0.0, -logits), np.logaddexp(0.0, logits), labels, weights, prior
    )


class PURiskMetric:
    def __init__(self, prior: float, *, non_negative: bool) -> None:
        self.prior = prior
        self.non_negative = non_negative

    def is_max_optimal(self) -> bool:
        return False

    def get_final_error(self, error: float, weight: float) -> float:
        return error / weight

    def evaluate(
        self,
        approxes: Sequence[Sequence[float]],
        target: Sequence[float],
        weight: Sequence[float] | None,
    ) -> tuple[float, float]:
        if len(approxes) != 1:
            raise ValueError("PU risk requires one-dimensional raw predictions")
        weights = np.ones(len(target)) if weight is None else weight
        positive_risk, negative_risk = pu_risks(
            np.asarray(approxes[0]), np.asarray(target), np.asarray(weights), self.prior
        )
        risk = positive_risk + (
            max(0.0, negative_risk) if self.non_negative else negative_risk
        )
        return risk, 1.0


class UPUMetric(PURiskMetric):
    no_jit = True

    def __init__(self, prior: float) -> None:
        super().__init__(prior, non_negative=False)

    def evaluate(
        self,
        approxes: Sequence[Sequence[float]],
        target: Sequence[float],
        weight: Sequence[float] | None,
    ) -> tuple[float, float]:
        from .cpu import upu_risk

        if len(approxes) != 1:
            raise ValueError("PU risk requires one-dimensional raw predictions")
        risk = upu_risk(
            np.asarray(approxes[0], dtype=np.float64),
            np.asarray(target),
            None if weight is None else np.asarray(weight),
            self.prior,
        )
        return risk, 1.0


class NNPUMetric(PURiskMetric):
    def __init__(self, prior: float) -> None:
        super().__init__(prior, non_negative=True)


class PULogisticObjective:
    def __init__(
        self,
        prior: float,
        positive_scale: float,
        unlabeled_scale: float,
        *,
        reverse_negative_risk: bool = False,
    ) -> None:
        self.prior = prior
        self.positive_scale = positive_scale
        self.unlabeled_scale = unlabeled_scale
        self.reverse_negative_risk = reverse_negative_risk

    def calc_ders_range(
        self,
        approxes: Sequence[float],
        targets: Sequence[float],
        weights: Sequence[float] | None,
    ) -> list[tuple[float, float]]:
        result: list[tuple[float, float]] = []
        for index in range(len(targets)):
            logit = float(approxes[index])
            if logit >= 0:
                probability = 1.0 / (1.0 + math.exp(-logit))
            else:
                exp_logit = math.exp(logit)
                probability = exp_logit / (1.0 + exp_logit)
            weight = 1.0 if weights is None else float(weights[index])
            curvature = probability * (1.0 - probability)

            if targets[index] == 1:
                if self.reverse_negative_risk:
                    # -R_negative has derivative +prior * sigmoid(logit).
                    der1 = -self.positive_scale * probability
                    der2 = -self.positive_scale * curvature
                else:
                    # Logistic uPU: loss_+(logit) - loss_-(logit) = -logit.
                    der1 = self.positive_scale
                    der2 = 0.0
            elif targets[index] == 0:
                direction = 1.0 if self.reverse_negative_risk else -1.0
                der1 = direction * self.unlabeled_scale * probability
                der2 = direction * self.unlabeled_scale * curvature
            else:
                raise ValueError("PU labels must be 1 or 0")
            result.append((weight * der1, weight * der2))
        return result
