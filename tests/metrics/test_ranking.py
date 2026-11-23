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
    assert result.gains_area == pytest.approx(43 / 72)
    assert result.normalized_gini == pytest.approx(1)
    assert [point.percent for point in result.selected_points] == [75, 25, 100]
    np.testing.assert_allclose(
        [p.gain for p in result.selected_points], [8 / 9, 1 / 3, 1]
    )
    np.testing.assert_allclose(
        [p.lift for p in result.selected_points], [32 / 27, 4 / 3, 1]
    )


def test_ties_form_one_step_and_constant_targets_have_undefined_normalized_gini():
    tied = ranking_metrics(np.array([1.0, 3.0]), np.ones(2), np.ones(2))
    np.testing.assert_allclose(tied.cumulative_exposure, [0, 1])
    np.testing.assert_allclose(tied.cumulative_observed, [0, 1])
    constant = ranking_metrics(np.ones(2), np.array([0.2, 0.8]), np.array([1.0, 3.0]))
    assert constant.gini == pytest.approx(0)
    assert constant.gains_area == pytest.approx(0.5)
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


@pytest.mark.parametrize("order", [[0, 1, 2, 3], [3, 2, 1, 0], [0, 2, 1, 3]])
def test_constant_prediction_has_zero_gini_for_any_row_order(order):
    y = np.array([0.0, 0.0, 1.0, 1.0])[order]
    result = ranking_metrics(y, np.full(4, 0.3), np.ones(4))
    assert result.gini == pytest.approx(0)
    assert result.normalized_gini == pytest.approx(0)


def test_heavy_ties_are_invariant_to_row_permutation():
    rng = np.random.default_rng(0)
    n = 200
    y = rng.gamma(1.0, 1.0, n)
    score = np.round(rng.random(n), 1)
    w = rng.uniform(0.5, 2.0, n)
    base = ranking_metrics(y, score, w, selected_percents=(5, 33, 50))
    perm = rng.permutation(n)
    other = ranking_metrics(
        y[perm], score[perm], w[perm], selected_percents=(5, 33, 50)
    )
    assert other.gini == pytest.approx(base.gini)
    assert other.normalized_gini == pytest.approx(base.normalized_gini)
    assert other.gains_area == pytest.approx(base.gains_area)
    np.testing.assert_allclose(other.cumulative_exposure, base.cumulative_exposure)
    np.testing.assert_allclose(other.cumulative_observed, base.cumulative_observed)
    np.testing.assert_allclose(
        [p.gain for p in other.selected_points], [p.gain for p in base.selected_points]
    )


def test_untied_data_matches_previous_implementation():
    # Expected values were captured from the pre-change implementation.
    result = ranking_metrics(
        np.array([1.0, 4.0, 2.0, 3.0, 0.5]),
        np.array([0.1, 0.8, 0.9, 0.5, 0.3]),
        np.array([1.0, 2.0, 1.0, 2.0, 3.0]),
        selected_percents=(35.0,),
    )
    assert result.gini == pytest.approx(0.28828828828828845, abs=1e-12)
    assert result.normalized_gini == pytest.approx(0.7619047619047625, abs=1e-12)
    assert result.gains_area == pytest.approx(0.6441441441441442, abs=1e-12)
    assert result.perfect_gini == pytest.approx(0.3783783783783783, abs=1e-12)
    assert result.selected_points[0].gain == pytest.approx(
        0.5648648648648649, abs=1e-12
    )
    assert len(result.cumulative_observed) == 6


def test_selected_gain_inside_tie_block_interpolates_linearly():
    # Scores: 0.9 (row 0) then a tie block of rows 1-2 covering exposure 1..4 of 4.
    result = ranking_metrics(
        np.array([2.0, 0.0, 2.0]),
        np.array([0.9, 0.5, 0.5]),
        np.array([1.0, 1.0, 2.0]),
        selected_percents=(50.0,),
    )
    np.testing.assert_allclose(result.cumulative_exposure, [0, 0.25, 1])
    np.testing.assert_allclose(result.cumulative_observed, [0, 1 / 3, 1])
    assert result.selected_points[0].gain == pytest.approx(
        1 / 3 + (0.5 - 0.25) / 0.75 * 2 / 3
    )


def test_tied_curve_hand_computed_values():
    # Sorted: score 0.9 (y=3, w=1), then tie of (y=1, w=1) and (y=0, w=2).
    # observed = 3, 1, 0 -> total 4; exposure total 4.
    result = ranking_metrics(
        np.array([1.0, 3.0, 0.0]),
        np.array([0.5, 0.9, 0.5]),
        np.array([1.0, 1.0, 2.0]),
    )
    np.testing.assert_allclose(result.cumulative_exposure, [0, 0.25, 1])
    np.testing.assert_allclose(result.cumulative_observed, [0, 0.75, 1])
    np.testing.assert_allclose(result.cumulative_lift[1:], [3.0, 1.0])
    assert np.isnan(result.cumulative_lift[0])


def test_zero_exposure_row_inside_tie_block_is_ignored():
    y = np.array([1.0, 3.0, 5.0, 0.0])
    score = np.array([0.5, 0.9, 0.5, 0.5])
    w = np.array([1.0, 1.0, 2.0, 1.0])
    with_zero = ranking_metrics(y, score, np.array([1.0, 1.0, 0.0, 2.0]))
    without = ranking_metrics(
        np.array([1.0, 3.0, 0.0]), np.array([0.5, 0.9, 0.5]), np.array([1.0, 1.0, 2.0])
    )
    assert w.sum() == 5.0
    np.testing.assert_allclose(
        with_zero.cumulative_exposure, without.cumulative_exposure
    )
    np.testing.assert_allclose(
        with_zero.cumulative_observed, without.cumulative_observed
    )
    assert with_zero.gini == pytest.approx(without.gini)
    assert with_zero.perfect_gini == pytest.approx(without.perfect_gini)


def test_perfect_gini_with_repeated_targets_is_hand_computed():
    # Targets 2, 2, 0, 0 with unit exposure: perfect curve steps at the
    # tied 2s (0.5, 1.0) and the tied 0s (1.0): points (0,0),(0.5,1),(1,1).
    # Area = 0.75, perfect gini = 0.5. Prediction order is the 2, 0, 2, 0.
    result = ranking_metrics(
        np.array([2.0, 2.0, 0.0, 0.0]),
        np.array([0.9, 0.3, 0.5, 0.1]),
        np.ones(4),
    )
    assert result.perfect_gini == pytest.approx(0.5)
    # Model curve: exposure 0,.25,.5,.75,1; observed 0,.5,.5,1,1 -> area 0.625.
    assert result.gains_area == pytest.approx(0.625)
    assert result.gini == pytest.approx(0.25)
    assert result.normalized_gini == pytest.approx(0.5)
