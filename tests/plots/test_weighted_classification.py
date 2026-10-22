import numpy as np
import pandas as pd
import pytest
from ml_toolkit.plots.classification import (
    binary_curves,
    calibration,
    precision_recall,
    f1_threshold,
)


def test_weighted_calibration_and_plots(tmp_path):
    y, p, w = (
        np.array([0.0, 1.0, 1.0]),
        np.array([0.2, 0.8, 0.9]),
        np.array([3.0, 1.0, 0.0]),
    )
    table = calibration(y, p, tmp_path / "calibration.png", weight=w, n_bins=1)
    assert table.observed_rate.iloc[0] == 0.25
    assert table.predicted_rate.iloc[0] == pytest.approx(0.35)
    assert table.weight_sum.iloc[0] == 4
    for draw in (precision_recall, f1_threshold):
        result = draw(y, p, tmp_path / f"{draw.__name__}.png", weight=w)
        pd.testing.assert_frame_equal(result, binary_curves(y, p, weight=w))


def test_weighted_plot_geometry_and_summary_labels(monkeypatch, tmp_path):
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score
    from ml_toolkit.plots import classification

    y, p, w = (
        np.array([1.0, 0.0, 1.0, 0.0]),
        np.array([0.9, 0.7, 0.4, 1.0]),
        np.array([5.0, 20.0, 1.0, 0.0]),
    )
    figures = []
    monkeypatch.setattr(
        classification, "save_figure", lambda fig, *a, **k: figures.append(fig)
    )
    try:
        precision_recall(y, p, tmp_path / "pr.png", weight=w)
        ax = figures[-1].axes[0]
        assert (
            ax.lines[0].get_label()
            == f"Model (weighted AP={average_precision_score(y, p, sample_weight=w):.3f})"
        )
        np.testing.assert_allclose(ax.lines[1].get_ydata(), np.average(y, weights=w))
        calibration(y, p, tmp_path / "cal.png", weight=w)
        histogram = figures[-1].axes[1]
        assert sum(patch.get_height() for patch in histogram.patches) == w.sum()
        assert "weight" in histogram.get_ylabel().lower()
    finally:
        for fig in figures:
            plt.close(fig)
