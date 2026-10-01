"""Evaluation of positive–unlabeled scores without treating U as negative."""

from dataclasses import dataclass

import numpy as np


def _inputs(labels, probabilities, weights):
    y = np.asarray(labels).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float).reshape(-1)
    if not len(y) or not (y.shape == p.shape == w.shape):
        raise ValueError("PU labels, scores, and weights need matching nonempty lengths")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("PU labels must be 1 (known positive) or 0 (unlabeled)")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("PU scores must be finite probabilities between 0 and 1")
    if not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("PU weights must be finite and nonnegative")
    if w[y == 1].sum() <= 0 or w[y == 0].sum() <= 0:
        raise ValueError("PU evaluation needs positive weight in both groups")
    return y, p, w


def positive_capture_curve(labels, probabilities, weights=None):
    """Return (fraction of U flagged, fraction of known P flagged) at score ties."""
    y, p, w = _inputs(labels, probabilities, weights)
    order = np.argsort(-p, kind="stable")
    y, p, w = y[order], p[order], w[order]
    ends = np.r_[np.flatnonzero(np.diff(p) != 0), len(p) - 1]
    u = np.cumsum(w * (y == 0))[ends] / w[y == 0].sum()
    capture = np.cumsum(w * (y == 1))[ends] / w[y == 1].sum()
    return np.r_[0.0, u], np.r_[0.0, capture]


@dataclass(frozen=True)
class PUMetric:
    """Callable metric for the training workflow; prior is population prevalence.

    Capture at capacity uses a score cutoff set by the unlabeled population.
    Ties at that cutoff are all included, so the actual capacity can be larger.
    Known-positive capture is a recall proxy only under representative labeling.
    """

    prior: float
    loss: str = "nnPU"

    def __post_init__(self):
        if not np.isfinite(self.prior) or not 0 < self.prior < 1:
            raise ValueError("PU prior must be strictly between 0 and 1")
        if self.loss not in ("uPU", "nnPU"):
            raise ValueError("PU loss must be 'uPU' or 'nnPU'")

    def __call__(self, labels, probabilities, weights=None) -> dict[str, float]:
        y, p, w = _inputs(labels, probabilities, weights)
        positive = y == 1
        unlabeled = ~positive
        clipped = np.clip(p, np.finfo(float).eps, 1 - np.finfo(float).eps)
        positive_risk = self.prior * np.average(-np.log(clipped[positive]), weights=w[positive])
        negative_risk = (
            np.average(-np.log1p(-clipped[unlabeled]), weights=w[unlabeled])
            - self.prior * np.average(-np.log1p(-clipped[positive]), weights=w[positive])
        )
        result = {
            "positive_risk": float(positive_risk),
            "estimated_negative_risk": float(negative_risk),
            "pu_risk": float(positive_risk + (max(0.0, negative_risk) if self.loss == "nnPU" else negative_risk)),
        }
        order = np.argsort(-p[unlabeled], kind="stable")
        ranked_scores = p[unlabeled][order]
        ranked_weights = w[unlabeled][order]
        cumulative_weight = np.cumsum(ranked_weights)
        for percent in (1, 5, 10):
            # The first score that reaches capacity sets the threshold.
            cutoff_index = int(np.searchsorted(
                cumulative_weight, cumulative_weight[-1] * percent / 100, side="left"
            ))
            cutoff = ranked_scores[cutoff_index]
            result[f"known_positive_capture_at_{percent}pct"] = float(
                w[positive & (p >= cutoff)].sum() / w[positive].sum()
            )
            result[f"unlabeled_flagged_at_{percent}pct"] = float(
                w[unlabeled & (p >= cutoff)].sum() / w[unlabeled].sum()
            )
        return result
