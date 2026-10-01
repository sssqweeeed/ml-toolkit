from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score

@dataclass(frozen=True)
class MetricsClassification:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    average_precision: float
    log_loss: float

def metrics_classification(
    y_target: np.ndarray,
    y_prob: np.ndarray,
    sample_weight: np.ndarray | None = None,
    *,
    threshold: float = 0.5,
) -> MetricsClassification:
    """Evaluate binary labels and predicted positive-class probabilities."""
    target = np.asarray(y_target).reshape(-1)
    probability = np.asarray(y_prob, dtype=float).reshape(-1)
    weight = None if sample_weight is None else np.asarray(sample_weight, dtype=float).reshape(-1)
    if target.shape != probability.shape or (weight is not None and weight.shape != target.shape):
        raise ValueError("Target, probabilities, and weights must have equal lengths")
    if not np.isin(target, [0, 1]).all():
        raise ValueError("Classification target must contain only 0 and 1")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Probabilities must be finite and between 0 and 1")
    if weight is not None and (not np.isfinite(weight).all() or (weight < 0).any()):
        raise ValueError("Weights must be finite and non-negative")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")

    labels = probability >= threshold
    has_both_classes = np.unique(target).size == 2
    return MetricsClassification(
        accuracy=float(accuracy_score(target, labels, sample_weight=weight)),
        precision=float(precision_score(target, labels, sample_weight=weight, zero_division=0)),
        recall=float(recall_score(target, labels, sample_weight=weight, zero_division=0)),
        f1=float(f1_score(target, labels, sample_weight=weight, zero_division=0)),
        roc_auc=float(roc_auc_score(target, probability, sample_weight=weight)) if has_both_classes else float("nan"),
        average_precision=float(average_precision_score(target, probability, sample_weight=weight)) if (target == 1).any() else 0.0,
        log_loss=float(log_loss(target, probability, sample_weight=weight, labels=[0, 1])),
    )

def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> tuple[float, float, float, float]:
    """Return weighted or unweighted binary counts as TN, FP, FN, TP."""
    target = np.asarray(y_true).reshape(-1)
    prediction = np.asarray(y_pred).reshape(-1)
    weight = np.ones(len(target)) if sample_weight is None else np.asarray(sample_weight, dtype=float).reshape(-1)
    if not (target.shape == prediction.shape == weight.shape):
        raise ValueError("Target, predictions, and weights must have equal lengths")
    if not np.isin(target, [0, 1]).all() or not np.isin(prediction, [0, 1]).all():
        raise ValueError("Confusion matrix requires binary labels")
    return tuple(
        float(weight[(target == actual) & (prediction == predicted)].sum())
        for actual, predicted in ((0, 0), (0, 1), (1, 0), (1, 1))
    )
