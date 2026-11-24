"""Computation split from figures: tables match the figure functions' returns."""

from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.metrics import average_precision_score

from ml_toolkit.metrics.classification import average_precision
from ml_toolkit.reports.base import (
    expand_skipped_rows,
    long_metric_rows,
    metric_scores,
)
from ml_toolkit.reports.classification import (
    binary_curves,
    calibration,
    calibration_bins,
    precision_recall,
    precision_recall_summary,
)
from ml_toolkit.reports.distributions import (
    distribution_summary,
    time_feature_distribution,
)
from ml_toolkit.reports.regression import (
    level_observed_fitted,
    observed_fitted_by_level,
    time_observed_fitted,
    time_summary,
)

RNG = np.random.default_rng(3)
LABELS = (RNG.random(60) < 0.4).astype(float)
PROBABILITIES = np.clip(0.3 * LABELS + 0.7 * RNG.random(60), 0, 1)
WEIGHTS = RNG.random(60) + 0.1
WEIGHTS[:5] = 0.0


def test_precision_recall_summary_matches_previous_step_sum_average_precision():
    summary = precision_recall_summary(LABELS, PROBABILITIES, weight=WEIGHTS)
    curve = binary_curves(LABELS, PROBABILITIES, weight=WEIGHTS)
    previous = float((np.diff(np.r_[0.0, curve.recall]) * curve.precision).sum())
    assert summary.average_precision == pytest.approx(previous, abs=1e-12)
    assert summary.average_precision == pytest.approx(
        average_precision_score(LABELS, PROBABILITIES, sample_weight=WEIGHTS)
    )
    assert summary.prevalence == pytest.approx(np.average(LABELS, weights=WEIGHTS))
    pd.testing.assert_frame_equal(summary.curve, curve)
    assert average_precision(LABELS, PROBABILITIES, WEIGHTS) == pytest.approx(
        previous, abs=1e-12
    )


def test_average_precision_is_zero_without_weighted_positives():
    labels = np.array([1.0, 0.0, 0.0])
    summary = precision_recall_summary(labels, [0.9, 0.5, 0.1], weight=[0.0, 1.0, 1.0])
    assert summary.average_precision == 0.0
    assert summary.prevalence == 0.0


def test_calibration_bins_equal_the_figure_function_table(tmp_path):
    binned = calibration_bins(LABELS, PROBABILITIES, 5, weight=WEIGHTS)
    drawn = calibration(
        LABELS, PROBABILITIES, tmp_path / "c.png", weight=WEIGHTS, n_bins=5
    )
    pd.testing.assert_frame_equal(binned.table, drawn)
    assert len(binned.probabilities) == int((WEIGHTS > 0).sum())
    assert binned.weight.sum() == pytest.approx(WEIGHTS.sum())


def test_precision_recall_returns_the_summary_curve(tmp_path):
    drawn = precision_recall(LABELS, PROBABILITIES, tmp_path / "p.png", weight=WEIGHTS)
    pd.testing.assert_frame_equal(
        drawn, precision_recall_summary(LABELS, PROBABILITIES, weight=WEIGHTS).curve
    )


def test_level_summary_equals_the_figure_function_table(tmp_path):
    order = ("a", "b", "c")
    reference = pl.Series(["a", "b", "a", "c", "b", "a"])
    current = pl.Series(["a", "b", "b", "a"])
    arguments = (
        reference,
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [1.5, 2.5, 2.0, 4.5, 4.0, 6.5],
        current,
        [2.0, 3.0, 1.0, 2.0],
        [2.5, 2.0, 1.5, 2.5],
    )
    table = level_observed_fitted(
        *arguments, level_order=order, reference_weight=[1, 2, 1, 3, 1, 1]
    )
    drawn = observed_fitted_by_level(
        *arguments,
        tmp_path / "levels.png",
        level_order=order,
        reference_weight=[1, 2, 1, 3, 1, 1],
    )
    pd.testing.assert_frame_equal(table, drawn)


def test_time_summary_equals_the_figure_function_table(tmp_path):
    times = [datetime(2024, month, 5) for month in (1, 1, 2, 2, 3, 3)]
    later = [datetime(2024, month, 5) for month in (3, 4, 4, 5)]
    arguments = (
        times,
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [1.5, 2.5, 2.0, 4.5, 4.0, 6.5],
        later,
        [2.0, 3.0, 1.0, 2.0],
        [2.5, 2.0, 1.5, 2.5],
    )
    summary = time_summary(*arguments, period="month")
    drawn = time_observed_fitted(*arguments, tmp_path / "t.png", period="month")
    pd.testing.assert_frame_equal(summary.table, drawn)
    assert summary.boundary == pd.Timestamp(2024, 3, 1)


def test_distribution_summary_equals_the_figure_function_table(tmp_path):
    def frame(months: list[int], values: list[float]) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "x": values,
                "time": [datetime(2024, month, 3) for month in months],
            }
        )

    reference = frame([1, 1, 1, 2, 2, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 9.0])
    current = frame([3, 3, 3, 4], [2.0, 6.0, float("nan"), 7.0])
    summary = distribution_summary(reference, current, "x", "time")
    drawn = time_feature_distribution(
        reference, current, "x", "time", tmp_path / "d.png"
    )
    assert summary.equals(drawn)
    assert summary.columns == [
        "period_start",
        "split",
        "p05",
        "p25",
        "p50",
        "mean",
        "p75",
        "p95",
    ]


def test_metric_rows_mark_nonfinite_scores_undefined_and_expand_skipped_rows():
    rows = long_metric_rows({"split": "train"}, {"mae": 1.0, "r2": float("nan")})
    assert [(row["metric"], row["value"], row["status"]) for row in rows] == [
        ("mae", 1.0, "ok"),
        ("r2", None, "undefined"),
    ]
    skipped = {"split": "val", "metric": None, "value": None, "status": "empty"}
    expanded = expand_skipped_rows([*rows, skipped], {"r2", "mae"})
    assert [row["metric"] for row in expanded[-2:]] == ["mae", "r2"]


@pytest.mark.parametrize("scores", [{}, {" ": 1.0}])
def test_metric_scores_require_named_scores(scores):
    with pytest.raises(ValueError, match="nonempty named scores"):
        metric_scores(scores)
