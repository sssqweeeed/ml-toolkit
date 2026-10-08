from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..types import FloatArray


@dataclass(frozen=True)
class MetricsClassification:
    """Store weighted binary classification scores.

    Attributes
    ----------
    accuracy : float
        Share of correct thresholded predictions.
    precision : float
        Positive predictive value; zero when undefined.
    recall : float
        Share of positives recovered; zero when undefined.
    f1 : float
        Harmonic mean of precision and recall.
    roc_auc : float
        ROC area from probability scores; NaN when only one target class is
        present.
    average_precision : float
        Average precision from scores; zero when no positives are present.
    log_loss : float
        Binary cross-entropy calculated from probabilities.
    """

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    average_precision: float
    log_loss: float

    def to_dict(self) -> dict[str, float]:
        """Return named scalar scores.

        Returns
        -------
        dict[str, float]
            All scores in this result.
        """
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "average_precision": self.average_precision,
            "log_loss": self.log_loss,
        }


def metrics_classification(
    y_target: FloatArray,
    y_prob: FloatArray,
    sample_weight: FloatArray | None = None,
    *,
    threshold: float = 0.5,
) -> MetricsClassification:
    """Evaluate binary labels and predicted positive-class probabilities.

    Parameters
    ----------
    y_target : numpy.ndarray of shape (n_samples,)
        Observed binary labels, containing only 0 and 1.
    y_prob : numpy.ndarray of shape (n_samples,)
        Finite positive-class probabilities in [0, 1].
    sample_weight : numpy.ndarray or None, default None
        Finite, nonnegative weights aligned with labels. None uses equal
        weights.
    threshold : float, default 0.5
        Probability cutoff in [0, 1]; scores at or above it predict label 1.

    Returns
    -------
    MetricsClassification
        Threshold-based accuracy, precision, recall, and F1, plus score-based
        ROC AUC, average precision, and log loss.

    Raises
    ------
    ValueError
        Labels, probabilities, weights, lengths, or threshold are invalid.

    Notes
    -----
    ROC AUC is NaN for a single target class. Precision and recall use zero for
    undefined divisions; average precision is zero with no positives. These
    metrics require binary ground truth; do not treat PU unlabeled rows as
    confirmed negatives.

    Examples
    --------
    >>> import numpy as np
    >>> from ml_toolkit.metrics import metrics_classification
    >>> scores = metrics_classification(np.array([0, 1]), np.array([0.2, 0.8]))
    >>> scores.accuracy, scores.f1
    (1.0, 1.0)
    """
    target = np.asarray(y_target).reshape(-1)
    probability = np.asarray(y_prob, dtype=float).reshape(-1)
    weight = (
        None
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float).reshape(-1)
    )
    if target.shape != probability.shape or (
        weight is not None and weight.shape != target.shape
    ):
        raise ValueError("Target, probabilities, and weights must have equal lengths")
    if not np.isin(target, [0, 1]).all():
        raise ValueError("Classification target must contain only 0 and 1")
    if (
        not np.isfinite(probability).all()
        or ((probability < 0) | (probability > 1)).any()
    ):
        raise ValueError("Probabilities must be finite and between 0 and 1")
    if weight is not None and (not np.isfinite(weight).all() or (weight < 0).any()):
        raise ValueError("Weights must be finite and non-negative")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")

    labels = probability >= threshold
    has_both_classes = np.unique(target).size == 2
    return MetricsClassification(
        accuracy=float(accuracy_score(target, labels, sample_weight=weight)),
        precision=float(
            precision_score(target, labels, sample_weight=weight, zero_division=0)
        ),
        recall=float(
            recall_score(target, labels, sample_weight=weight, zero_division=0)
        ),
        f1=float(f1_score(target, labels, sample_weight=weight, zero_division=0)),
        roc_auc=(
            float(roc_auc_score(target, probability, sample_weight=weight))
            if has_both_classes
            else float("nan")
        ),
        average_precision=(
            float(average_precision_score(target, probability, sample_weight=weight))
            if (target == 1).any()
            else 0.0
        ),
        log_loss=float(
            log_loss(target, probability, sample_weight=weight, labels=[0, 1])
        ),
    )


def confusion_matrix(
    y_true: FloatArray,
    y_pred: FloatArray,
    sample_weight: FloatArray | None = None,
) -> tuple[float, float, float, float]:
    """Return weighted binary counts in TN, FP, FN, TP order.

    Parameters
    ----------
    y_true : numpy.ndarray of shape (n_samples,)
        Observed binary labels.
    y_pred : numpy.ndarray of shape (n_samples,)
        Predicted binary labels, rather than probability scores.
    sample_weight : numpy.ndarray or None, default None
        Row weights aligned with labels; None counts every row as one.

    Returns
    -------
    tuple[float, float, float, float]
        True negatives, false positives, false negatives, and true positives.

    Raises
    ------
    ValueError
        Lengths differ or labels are not binary.

    Examples
    --------
    >>> from ml_toolkit.metrics import confusion_matrix
    >>> confusion_matrix([0, 0, 1, 1], [0, 1, 0, 1])
    (1.0, 1.0, 1.0, 1.0)
    """
    target = np.asarray(y_true).reshape(-1)
    prediction = np.asarray(y_pred).reshape(-1)
    weight = (
        np.ones(len(target))
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float).reshape(-1)
    )
    if not (target.shape == prediction.shape == weight.shape):
        raise ValueError("Target, predictions, and weights must have equal lengths")
    if not np.isin(target, [0, 1]).all() or not np.isin(prediction, [0, 1]).all():
        raise ValueError("Confusion matrix requires binary labels")
    tn, fp, fn, tp = (
        float(weight[(target == actual) & (prediction == predicted)].sum())
        for actual, predicted in ((0, 0), (0, 1), (1, 0), (1, 1))
    )
    return tn, fp, fn, tp
