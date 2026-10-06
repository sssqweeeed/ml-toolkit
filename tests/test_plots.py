from datetime import date

import matplotlib
import numpy as np
import polars as pl
import pytest

from ml_toolkit.config import (
    ClassificationPlotsConfig,
)

matplotlib.use("Agg")

from ml_toolkit.plots.classification import (
    binary_curves,
    calibration,
    f1_threshold,
    precision_recall,
)
from ml_toolkit.plots.explainability import feature_importance
from ml_toolkit.plots.monitoring import monthly_feature_distribution
from ml_toolkit.plots.ranking import gains_curve, lift_curve
from ml_toolkit.metrics.ranking import ranking_metrics as ranking_curve
from ml_toolkit.plots.regression import monthly_observed_fitted, observed_fitted
from ml_toolkit.plots.utils import PlotError


def test_binary_curves_and_plots(tmp_path):
    y, p = [0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8]
    curve = binary_curves(y, p)
    assert curve.iloc[1].precision == 1
    assert curve.iloc[1].recall == 1
    assert curve.iloc[1].f1 == 1
    assert precision_recall(y, p, tmp_path / "pr.png").equals(curve)
    assert f1_threshold(y, p, tmp_path / "f1.png").equals(curve)
    bins = calibration(
        y,
        p,
        tmp_path / "calibration.png",
        config=ClassificationPlotsConfig(calibration_bins=2),
    )
    assert bins["count"].sum() == 4
    assert all(
        (tmp_path / name).is_file() for name in ["pr.png", "f1.png", "calibration.png"]
    )


def test_classification_rejects_invalid_probabilities():
    with pytest.raises(PlotError):
        binary_curves([0, 1], [0.2, 1.2])


def test_ranking_curve_and_figures(tmp_path):
    curve = ranking_curve(
        [0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2], [1, 1, 1, 1], selected_percents=(50,)
    )
    assert curve.selected_points[0].gain == 1
    assert curve.selected_points[0].lift == 2
    assert np.isclose(curve.normalized_gini, 1)
    gains_curve(curve, tmp_path / "gains.png")
    lift_curve(curve, tmp_path / "lift.png")
    assert (tmp_path / "gains.png").is_file() and (tmp_path / "lift.png").is_file()


def test_regression_weighted_aggregations(tmp_path):
    summary = observed_fitted(
        ["a", "a", "b"],
        [1, 3, 4],
        [2, 2, 5],
        tmp_path / "feature.png",
        weight=[1, 3, 1],
    )
    assert summary.loc[summary.feature == "a", "observed"].iloc[0] == 2.5
    monthly = monthly_observed_fitted(
        ["2025-01-01", "2025-01-20"],
        [1, 3],
        [2, 2],
        ["2025-02-01"],
        [4],
        [5],
        tmp_path / "monthly.png",
        reference_weight=[1, 3],
        reference_group=["a", "a"],
        current_group=["a"],
    )
    assert monthly.loc[monthly.split == "reference", "observed"].iloc[0] == 2.5
    assert (tmp_path / "feature.png").is_file() and (tmp_path / "monthly.png").is_file()


def test_monthly_numeric_distribution_and_importance(tmp_path):
    reference = pl.DataFrame(
        {"date": [date(2025, 1, 1), date(2025, 1, 15)], "value": [1.0, 3.0]}
    )
    current = pl.DataFrame(
        {"date": [date(2025, 2, 1), date(2025, 2, 15)], "value": [2.0, 4.0]}
    )
    summary = monthly_feature_distribution(
        reference, current, "value", "date", tmp_path / "feature.png"
    )
    assert summary.height == 2
    assert summary.filter(pl.col("split") == "reference").get_column("mean").item() == 2
    ordered = feature_importance(["a", "b"], [0.2, 0.8], tmp_path / "importance.png")
    assert ordered.feature.tolist() == ["b", "a"]
    assert (tmp_path / "feature.png").is_file() and (
        tmp_path / "importance.png"
    ).is_file()
