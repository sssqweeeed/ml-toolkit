"""Ranking capacity follows exposure, including interpolation and zero-weight rows."""

import numpy as np
import pytest

from ml_toolkit.metrics import ranking_metrics


def test_weighted_gain_curve_interpolation_and_requested_point_order():
    result = ranking_metrics(
        np.array([1.0, 3.0, 2.0, 999.0]),
        np.array([0.1, 0.9, 0.5, 1.0]),
        np.array([1.0, 2.0, 1.0, 0.0]),
        selected_percents=(75, 25, 100),
    )
    np.testing.assert_allclose(result.cumulative_exposure, [0, 0.5, 0.75, 1])
    np.testing.assert_allclose(result.cumulative_observed, [0, 6 / 9, 8 / 9, 1])
    assert np.isnan(result.cumulative_lift[0])
    assert result.gini == pytest.approx(7 / 36)
    assert result.normalized_gini == pytest.approx(1)
    assert [point.percent for point in result.selected_points] == [75, 25, 100]
    np.testing.assert_allclose([p.gain for p in result.selected_points], [8 / 9, 1 / 3, 1])
    np.testing.assert_allclose([p.lift for p in result.selected_points], [32 / 27, 4 / 3, 1])


def test_ties_preserve_input_order_and_constant_targets_have_undefined_normalized_gini():
    tied = ranking_metrics(np.array([1., 3.]), np.ones(2), np.ones(2))
    np.testing.assert_allclose(tied.cumulative_observed, [0, 0.25, 1])
    constant = ranking_metrics(np.ones(2), np.array([0.2, 0.8]), np.array([1., 3.]))
    assert constant.gini == pytest.approx(0)
    assert np.isnan(constant.normalized_gini)


@pytest.mark.parametrize(
    "target, prediction, exposure, percent, message",
    [
        ([1, 2], [1], [1, 1], 10, "same shape"),
        ([1, np.nan], [1, 2], [1, 1], 10, "non-finite"),
        ([1, 2], [1, np.inf], [1, 1], 10, "non-finite"),
        ([1, 2], [1, 2], [1, np.nan], 10, "non-finite"),
        ([1, 2], [1, 2], [1, -1], 10, "negative"),
        ([1, 2], [1, 2], [0, 0], 10, "total_exposure"),
        ([0, 0], [1, 2], [1, 1], 10, "total_observed"),
        ([1, 2], [1, 2], [1, 1], 0, "selected percent"),
        ([1, 2], [1, 2], [1, 1], 101, "selected percent"),
    ],
)
def test_invalid_ranking_inputs_fail(target, prediction, exposure, percent, message):
    with pytest.raises(ValueError, match=message):
        ranking_metrics(
            np.asarray(target), np.asarray(prediction), np.asarray(exposure), (percent,)
        )
