"""Shared weighted reduction of PU loss components."""

import numpy as np

from ..types import FloatArray


def risk_components(
    positive_loss: FloatArray,
    negative_loss: FloatArray,
    labels: FloatArray,
    weights: FloatArray,
    prior: float,
) -> tuple[float, float]:
    positive = labels == 1
    unlabeled = ~positive
    positive_risk = prior * np.average(
        positive_loss[positive], weights=weights[positive]
    )
    negative_risk = np.average(
        negative_loss[unlabeled], weights=weights[unlabeled]
    ) - prior * np.average(negative_loss[positive], weights=weights[positive])
    return float(positive_risk), float(negative_risk)
