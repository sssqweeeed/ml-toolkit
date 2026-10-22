"""Independent arithmetic checks for shared regression scores."""

import numpy as np
import pytest

from ml_toolkit.metrics import metrics_regression, metrics_frequency, metrics_severity


def test_weighted_signed_errors_and_r2():
    # Weighted target mean = 0; SSE = 1 + 2*4 + 1 = 10; SST = 4 + 4 = 8.
    result = metrics_regression(
        np.array([-2.0, 0.0, 2.0, 100.0]),
        np.array([-1.0, -2.0, 3.0, -100.0]),
        np.array([1.0, 2.0, 1.0, 0.0]),
    )
    assert result.to_dict() == pytest.approx(
        {
            "mean_absolute_error": 1.5,
            "mean_squared_error": 2.5,
            "root_mean_squared_error": np.sqrt(2.5),
            "r2_score": -0.25,
            "mean_error": -0.5,
        }
    )


def test_unweighted_matches_unit_weights():
    y, p = np.array([-1.0, 0.0, 3.0]), np.array([0.0, 1.0, 2.0])
    assert metrics_regression(y, p) == metrics_regression(y, p, np.ones(3))


@pytest.mark.parametrize(
    "predicted,expected", [([2.0, 2.0, 9.0], 1.0), ([2.0, 3.0, 9.0], 0.0)]
)
def test_constant_target_ignores_zero_weight_rows(predicted, expected):
    result = metrics_regression(
        np.array([2.0, 2.0, -8.0]), np.array(predicted), np.array([1.0, 2.0, 0.0])
    )
    assert result.r2_score == expected


@pytest.mark.parametrize("weight", [np.array([1.0, 0.0]), np.array([0.0, 2.0])])
def test_r2_needs_two_positive_weight_rows(weight):
    result = metrics_regression(np.array([1.0, 2.0]), np.array([1.0, 3.0]), weight)
    assert np.isnan(result.r2_score)
    assert np.isfinite(result.mean_squared_error)


@pytest.mark.parametrize(
    "y,p,w",
    [
        ([], [], None),
        ([[1.0]], [[1.0]], None),
        ([1.0, 2.0], [1.0], None),
        ([1.0], [1.0], [1.0, 2.0]),
        ([1.0], [[1.0]], None),
        ([np.nan], [1.0], None),
        ([1.0], [np.inf], None),
        ([1.0], [1.0], [np.nan]),
        ([1.0], [1.0], [np.inf]),
        ([1.0], [1.0], [-1.0]),
        ([1.0], [1.0], [0.0]),
    ],
)
def test_invalid_regression_inputs(y, p, w):
    with pytest.raises(ValueError):
        metrics_regression(
            np.asarray(y), np.asarray(p), None if w is None else np.asarray(w)
        )


@pytest.mark.parametrize(
    "specialized",
    [metrics_frequency, lambda y, p, w: metrics_severity(y, p, w, power=2)],
)
def test_specialized_scores_include_all_general_metrics(specialized):
    y, p, w = (
        np.array([1.0, 2.0, 4.0]),
        np.array([2.0, 2.0, 3.0]),
        np.array([1.0, 2.0, 3.0]),
    )
    common = metrics_regression(y, p, w).to_dict()
    result = specialized(y, p, w)
    for name, value in common.items():
        assert getattr(result, name) == pytest.approx(value)
        assert result.to_dict()[name] == pytest.approx(value)
