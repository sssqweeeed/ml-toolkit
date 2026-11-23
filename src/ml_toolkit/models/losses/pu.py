"""Training-only PU objectives, risk metrics and the nnPU boosting loop.

Numba-compiled uPU derivatives and risk, CatBoost metric and objective classes,
and ``fit_nnpu``. Serving never imports this module.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import TypeAlias, cast

import catboost
import numpy as np
from numba import njit
from numpy.typing import NDArray

from ..._console import Logger
from ...data import Pool
from ...metrics.pu import risk_components
from ...types import FloatArray, ModelParams
from .._catboost import ITERATION_ALIASES

logger = Logger(__name__)

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


def pu_risks(
    logits: FloatArray,
    labels: FloatArray,
    weights: FloatArray,
    prior: float,
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
    # Python callbacks: skip CatBoost's numba compile attempt on every fit.
    no_jit = True

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
    def __init__(self, prior: float) -> None:
        super().__init__(prior, non_negative=False)

    def evaluate(
        self,
        approxes: Sequence[Sequence[float]],
        target: Sequence[float],
        weight: Sequence[float] | None,
    ) -> tuple[float, float]:
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
    # Python callbacks: skip CatBoost's numba compile attempt on every fit.
    no_jit = True

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


def _tree_logits(
    model: catboost.CatBoostClassifier, pool: catboost.Pool, tree: int
) -> FloatArray:
    # Raw contribution of one tree; the pool must not carry a baseline.
    return np.asarray(
        model.predict(
            pool,
            prediction_type="RawFormulaVal",
            ntree_start=tree,
            ntree_end=tree + 1,
        ),
        dtype=float,
    ).reshape(-1)


def fit_nnpu(
    *,
    train: Pool,
    validation: Pool | None,
    baseline: float,
    prior: float,
    objective: PULogisticObjective,
    params: ModelParams,
) -> tuple[catboost.CatBoostClassifier, int | None]:
    """Update full-training-set nnPU before each tree (beta=0, gamma=1)."""
    params = params.copy()
    iteration_names = [key for key in ITERATION_ALIASES if key in params]
    if len(iteration_names) > 1:
        raise ValueError("Specify only one CatBoost iteration parameter")
    iterations = (
        int(cast(int, params.pop(iteration_names[0]))) if iteration_names else 1000
    )
    if iterations < 1:
        raise ValueError("iterations must be positive")
    stopping_rounds = params.pop("early_stopping_rounds", None)
    if stopping_rounds is not None and int(cast(int, stopping_rounds)) < 1:
        raise ValueError("early_stopping_rounds must be positive")
    use_best_model = bool(params.pop("use_best_model", validation is not None))
    params["iterations"] = 1
    # CatBoost's automatic learning rate would otherwise use a 1-tree run.
    params.setdefault("learning_rate", 0.03)
    params.pop("verbose", None)
    params["logging_level"] = "Silent"
    train_data = train.to_catboost(baseline=baseline)
    validation_data = (
        None if validation is None else validation.to_catboost(baseline=baseline)
    )
    # Each tree's contribution is predicted on baseline-free pools, because
    # CatBoost adds a pool baseline to every prediction call.
    tree_train_data = train.to_catboost()
    tree_validation_data = None if validation is None else validation.to_catboost()
    train_logits = np.full(len(train.y), baseline, dtype=float)
    if validation is not None:
        pu_risks(
            np.full(len(validation.y), baseline),
            validation.y.to_numpy(),
            validation.weight.to_numpy(),
            prior,
        )
    train_labels = train.y.to_numpy()
    train_weights = train.weight.to_numpy()
    validation_labels = None if validation is None else validation.y.to_numpy()
    validation_weights = None if validation is None else validation.weight.to_numpy()
    validation_logits = (
        None if validation is None else np.full(len(validation.y), baseline)
    )
    model = None
    best_model = None
    best_iteration = None
    best_risk = float("inf")

    for iteration in range(iterations):
        # CatBoost's derivative callback sees only a subset, so compute
        # this decision from the complete training pool between trees.
        _, negative_risk = pu_risks(train_logits, train_labels, train_weights, prior)
        objective.reverse_negative_risk = negative_risk < 0.0
        tree_params = {**params, "loss_function": objective}
        next_model = catboost.CatBoostClassifier(**tree_params)
        next_model.fit(train_data, eval_set=validation_data, init_model=model)
        model = next_model
        train_logits += _tree_logits(model, tree_train_data, iteration)
        if iteration == 0 or (iteration + 1) % 50 == 0:
            logger.debug(
                "nnPU tree %d/%d; training negative risk %.6g; correction=%s",
                iteration + 1,
                iterations,
                negative_risk,
                objective.reverse_negative_risk,
            )

        if (
            tree_validation_data is not None
            and validation_logits is not None
            and validation_labels is not None
            and validation_weights is not None
        ):
            validation_logits += _tree_logits(model, tree_validation_data, iteration)
            positive_risk, negative_risk = pu_risks(
                validation_logits, validation_labels, validation_weights, prior
            )
            risk = positive_risk + max(0.0, negative_risk)
            if risk < best_risk:
                best_risk = risk
                best_iteration = iteration
                best_model = model
            if stopping_rounds is not None and iteration - cast(
                int, best_iteration
            ) >= int(cast(int, stopping_rounds)):
                break

    assert model is not None
    if validation is not None and use_best_model:
        assert best_model is not None
        model = best_model
    return model, best_iteration
