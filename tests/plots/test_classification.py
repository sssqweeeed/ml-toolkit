import pytest

from ml_toolkit.config import ClassificationPlotsConfig
from ml_toolkit.plots.classification import (
    binary_curves,
    calibration,
    f1_threshold,
    precision_recall,
)
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
