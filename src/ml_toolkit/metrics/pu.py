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
    """Calculate known-positive capture at unlabeled review capacities.

    Parameters
    ----------
    labels : array-like of shape (n_samples,)
        1 identifies a known positive; 0 identifies an unlabeled observation.
    probabilities : array-like of shape (n_samples,)
        Finite scores in [0, 1], aligned with labels.
    weights : array-like or None, default None
        Finite, nonnegative row weights; both groups must have positive total.
        None uses equal weights.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Weighted fraction of unlabeled rows flagged and weighted fraction of
        known positives captured at each distinct score cutoff. Both arrays
        start at zero.

    Raises
    ------
    ValueError
        Inputs are empty, misaligned, invalid, or a group has no positive
        weight.

    Notes
    -----
    All rows tied at a cutoff are included. Known-positive capture estimates
    recall only when labeled positives represent the full positive population.
    Unlabeled observations are not confirmed negatives.

    Examples
    --------
    >>> from ml_toolkit.metrics import positive_capture_curve
    >>> capacity, capture = positive_capture_curve([1, 0, 1, 0], [0.9, 0.8, 0.4, 0.2])
    >>> capacity.tolist(), capture.tolist()
    ([0.0, 0.0, 0.5, 0.5, 1.0], [0.0, 0.5, 0.5, 1.0, 1.0])
    """
    y, p, w = _inputs(labels, probabilities, weights)
    order = np.argsort(-p, kind="stable")
    y, p, w = y[order], p[order], w[order]
    ends = np.r_[np.flatnonzero(np.diff(p) != 0), len(p) - 1]
    u = np.cumsum(w * (y == 0))[ends] / w[y == 0].sum()
    capture = np.cumsum(w * (y == 1))[ends] / w[y == 1].sum()
    return np.r_[0.0, u], np.r_[0.0, capture]


@dataclass(frozen=True)
class PUMetric:
    """Evaluate positive-unlabeled risk without treating unlabeled rows as negative.

    Parameters
    ----------
    prior : float
        Finite population positive prevalence strictly between zero and one,
        rather than the labeled-positive fraction.
    loss : {'uPU', 'nnPU'}, default 'nnPU'
        uPU retains estimated negative risk; nnPU clips it at zero when
        computing total risk.

    Notes
    -----
    Callable metric for the training workflow. Match prior and loss to the
    model settings. Capture at capacity uses a score cutoff set by the
    unlabeled population. Ties at that cutoff are all included, so actual
    capacity can be larger. Known-positive capture is a recall proxy only under
    representative labeling.

    Examples
    --------
    >>> from ml_toolkit.metrics import PUMetric
    >>> scores = PUMetric(prior=0.1)([1, 0, 1, 0], [0.9, 0.8, 0.4, 0.2])
    >>> scores["known_positive_capture_at_10pct"]
    0.5
    """

    prior: float
    loss: str = "nnPU"

    def __post_init__(self):
        if not np.isfinite(self.prior) or not 0 < self.prior < 1:
            raise ValueError("PU prior must be strictly between 0 and 1")
        if self.loss not in ("uPU", "nnPU"):
            raise ValueError("PU loss must be 'uPU' or 'nnPU'")

    def __call__(self, labels, probabilities, weights=None) -> dict[str, float]:
        """Return PU risk and capture at 1%, 5%, and 10% unlabeled capacity.

        Parameters
        ----------
        labels : array-like of shape (n_samples,)
            Known positives are 1; unlabeled rows are 0.
        probabilities : array-like of shape (n_samples,)
            Finite scores in [0, 1]. Scores are clipped away from endpoints for
            log-risk calculations.
        weights : array-like or None, default None
            Finite, nonnegative weights with positive total in both groups; None
            uses equal weights.

        Returns
        -------
        dict[str, float]
            positive_risk, estimated_negative_risk, and pu_risk, plus
            known_positive_capture_at_{percent}pct and
            unlabeled_flagged_at_{percent}pct for percent in (1, 5, 10). Capacity
            and capture values are fractions in [0, 1].

        Raises
        ------
        ValueError
            Inputs are empty, invalid, misaligned, or lack positive weight in
            either group.
        """
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
