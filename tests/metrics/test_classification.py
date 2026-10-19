"""Weighted binary scores, threshold boundaries, and undefined classes."""

import numpy as np
import pytest

from ml_toolkit.metrics import confusion_matrix, metrics_classification


def test_weighted_scores_use_inclusive_threshold_and_probability_ranking():
    target = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.5, 0.4, 0.9])
    weight = np.array([1.0, 2.0, 3.0, 4.0])
    scores = metrics_classification(target, probability, weight)
    assert confusion_matrix(target, probability >= 0.5, weight) == (1, 2, 3, 4)
    assert scores.accuracy == pytest.approx(0.5)
    assert scores.precision == pytest.approx(4 / 6)
    assert scores.recall == pytest.approx(4 / 7)
    assert scores.f1 == pytest.approx(8 / 13)
    assert scores.roc_auc == pytest.approx(15 / 21)
    assert scores.average_precision == pytest.approx((4 + 3 * 7 / 9) / 7)
    assert scores.log_loss == pytest.approx(
        -(np.log(0.9) + 2 * np.log(0.5) + 3 * np.log(0.4) + 4 * np.log(0.9)) / 10
    )
    higher = metrics_classification(target, probability, weight, threshold=0.6)
    assert higher.precision == 1.0
    assert higher.roc_auc == scores.roc_auc
    assert higher.log_loss == scores.log_loss
    assert scores.to_dict()["f1"] == scores.f1


@pytest.mark.parametrize("label", [0, 1])
def test_single_class_scores_keep_defined_values_and_nan_auc(label):
    scores = metrics_classification(np.full(3, label), np.full(3, 0.25))
    assert np.isnan(scores.roc_auc)
    assert scores.average_precision == float(label)
    assert scores.precision == scores.recall == scores.f1 == 0
    assert np.isfinite(scores.log_loss)


@pytest.mark.parametrize(
    "target, probability, weight, threshold, message",
    [
        ([0, 1], [0.5], None, 0.5, "equal lengths"),
        ([0, 1], [0.2, 0.8], [1], 0.5, "equal lengths"),
        ([0, 2], [0.2, 0.8], None, 0.5, "only 0 and 1"),
        ([0, 1], [np.nan, 0.8], None, 0.5, "Probabilities"),
        ([0, 1], [0.2, 1.1], None, 0.5, "Probabilities"),
        ([0, 1], [0.2, 0.8], [1, -1], 0.5, "Weights"),
        ([0, 1], [0.2, 0.8], [1, np.inf], 0.5, "Weights"),
        ([0, 1], [0.2, 0.8], None, np.nan, "Threshold"),
    ],
)
def test_invalid_classification_inputs_fail(target, probability, weight, threshold, message):
    with pytest.raises(ValueError, match=message):
        metrics_classification(
            np.asarray(target), np.asarray(probability),
            None if weight is None else np.asarray(weight), threshold=threshold,
        )
