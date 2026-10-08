import numpy as np
import pytest

from ml_toolkit.metrics import PUMetric, positive_capture_curve


def test_pu_risk_and_capture_do_not_count_unlabeled_as_negative():
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    result = PUMetric(0.5)(labels, scores).to_dict()
    expected_positive = 0.5 * np.mean(-np.log(scores[:2]))
    expected_negative = np.mean(-np.log1p(-scores[2:])) - 0.5 * np.mean(
        -np.log1p(-scores[:2])
    )
    assert result["positive_risk"] == pytest.approx(expected_positive)
    assert result["estimated_negative_risk"] == pytest.approx(expected_negative)
    assert result["pu_risk"] == pytest.approx(
        expected_positive + max(0, expected_negative)
    )
    assert result["known_positive_capture_at_5pct"] == 1.0
    assert result["unlabeled_flagged_at_5pct"] == 0.5
    capacity, capture = positive_capture_curve(labels, scores)
    assert capacity[0] == capture[0] == 0.0
    assert capacity[-1] == capture[-1] == 1.0


def test_pu_metric_rejects_missing_group():
    with pytest.raises(ValueError, match="both groups"):
        PUMetric(0.5)([1, 1], [0.7, 0.8])
