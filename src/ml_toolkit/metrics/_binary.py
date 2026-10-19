"""Weighted binary threshold arithmetic shared by models and plots."""

import numpy as np

from ..types import FloatArray


def binary_threshold_scores(
    labels: FloatArray, probabilities: FloatArray, weight: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    if (
        labels.ndim != 1
        or probabilities.shape != labels.shape
        or weight.shape != labels.shape
        or not len(labels)
    ):
        raise ValueError(
            "Labels, probabilities and weights must be matching nonempty vectors"
        )
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Threshold tuning requires binary labels 0 and 1")
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError("Probabilities must be finite and between 0 and 1")
    if (
        not np.isfinite(weight).all()
        or (weight < 0).any()
        or not np.isfinite(weight.sum())
        or weight.sum() <= 0
    ):
        raise ValueError(
            "Weights must be finite, nonnegative, with positive finite total"
        )
    positive_weight = weight > 0
    y, p, w = (
        labels[positive_weight],
        probabilities[positive_weight],
        weight[positive_weight],
    )
    order = np.argsort(-p, kind="stable")
    y, p, w = y[order], p[order], w[order]
    ends = np.r_[np.flatnonzero(np.diff(p) != 0), len(p) - 1]
    tp = np.cumsum(y * w)[ends]
    predicted = np.cumsum(w)[ends]
    positives = float(np.sum(y * w))
    precision = tp / predicted
    recall = tp / positives if positives > 0 else np.zeros(len(ends))
    f1 = 2 * tp / (predicted + positives) if positives > 0 else np.zeros(len(ends))
    return p[ends], precision, recall, f1
