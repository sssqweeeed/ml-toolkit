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


@pytest.mark.parametrize("loss", ["uPU", "nnPU"])
def test_weighted_risk_is_invariant_to_weight_scale_and_clips_only_nnpu(loss):
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.95, 0.8, 0.1, 0.2])
    weights = np.array([1.0, 3.0, 2.0, 1.0])
    positive = 0.5 * np.average(-np.log(scores[:2]), weights=weights[:2])
    negative = np.average(-np.log1p(-scores[2:]), weights=weights[2:]) - 0.5 * np.average(
        -np.log1p(-scores[:2]), weights=weights[:2]
    )
    assert negative < 0
    metric = PUMetric(0.5, loss=loss)
    result = metric(labels, scores, weights).to_dict()
    assert result["positive_risk"] == pytest.approx(positive)
    assert result["estimated_negative_risk"] == pytest.approx(negative)
    assert result["pu_risk"] == pytest.approx(positive + (negative if loss == "uPU" else 0))
    assert metric(labels, scores, weights * 17).to_dict() == pytest.approx(result)


def test_capture_curve_includes_all_ties_and_ignores_input_order():
    labels = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.9, 0.2, 0.1])
    weights = np.array([1.0, 3.0, 3.0, 1.0])
    capacity, capture = positive_capture_curve(labels, scores, weights)
    np.testing.assert_allclose(capacity, [0.0, 0.75, 0.75, 1.0])
    np.testing.assert_allclose(capture, [0.0, 0.25, 1.0, 1.0])
    order = [3, 1, 0, 2]
    shuffled = positive_capture_curve(labels[order], scores[order], weights[order])
    np.testing.assert_allclose(shuffled, (capacity, capture))
    metrics = PUMetric(0.5)(labels, scores, weights).to_dict()
    assert metrics["known_positive_capture_at_5pct"] == 0.25
    assert metrics["unlabeled_flagged_at_5pct"] == 0.75


@pytest.mark.parametrize(
    "labels, scores, weights, message",
    [
        ([], [], None, "matching nonempty"),
        ([1, 0], [0.9], None, "matching nonempty"),
        ([1, 2], [0.9, 0.1], None, "labels"),
        ([1, 0], [np.nan, 0.1], None, "finite probabilities"),
        ([1, 0], [0.9, 1.1], None, "finite probabilities"),
        ([1, 0], [0.9, 0.1], [-1, 1], "nonnegative"),
        ([1, 0], [0.9, 0.1], [1, np.inf], "finite"),
        ([1, 0], [0.9, 0.1], [0, 1], "both groups"),
        ([1, 0], [0.9, 0.1], [1, 0], "both groups"),
    ],
)
@pytest.mark.parametrize(
    "evaluate", [PUMetric(0.5), positive_capture_curve], ids=["risk", "capture"]
)
def test_pu_metrics_reject_invalid_data(labels, scores, weights, message, evaluate):
    with pytest.raises(ValueError, match=message):
        evaluate(labels, scores, weights)
