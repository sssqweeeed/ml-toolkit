"""Plot rendering configuration and reproducible explanation sampling."""

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from tests.support.training import make_data
from ml_toolkit.config import (
    ClassificationPlotsConfig,
    FigureConfig,
    NumericalFeatureDistributionConfig,
    ShapConfig,
)
from ml_toolkit.explainability import calculate_shap_values
from ml_toolkit.models import CatBoostRegressionModel
from ml_toolkit.plots.classification import calibration, precision_recall
from ml_toolkit.plots.monitoring import time_feature_distribution
from ml_toolkit.plots.utils import save_figure


def test_custom_quantile_bands_and_hidden_curves(tmp_path):
    data = make_data()
    frame = data.X.with_columns(data.time)
    config = NumericalFeatureDistributionConfig(
        quantile_bands=((0.1, 0.9),), show_mean=False, show_median=False
    )
    summary = time_feature_distribution(
        frame.head(len(frame) // 2),
        frame.tail(len(frame) - len(frame) // 2),
        "x",
        "date",
        tmp_path / "quantiles.png",
        config=config,
    )
    assert (
        "p10" in summary.columns
        and "p90" in summary.columns
        and "p05" not in summary.columns
    )
    assert (tmp_path / "quantiles.png").is_file()


def test_figure_export_honors_format_dimensions_and_dpi(tmp_path, monkeypatch):
    fig, _ = plt.subplots()
    calls = []
    save = fig.savefig

    def capture(path, **kwargs):
        calls.append((tuple(fig.get_size_inches()), kwargs))
        return save(path, **kwargs)

    monkeypatch.setattr(fig, "savefig", capture)
    result = save_figure(
        fig,
        tmp_path / "plot.png",
        config=FigureConfig(figsize=(4, 3), dpi=72, format="svg"),
    )
    assert result == tmp_path / "plot.svg" and result.is_file()
    assert calls[0][0] == (4, 3) and calls[0][1]["dpi"] == 72
    assert not plt.fignum_exists(fig.number)


def test_classification_resolution_preserves_full_curve(tmp_path, monkeypatch):
    cfg = ClassificationPlotsConfig(
        max_curve_points=3, calibration_bins=2, histogram_bins=4
    )
    y = np.array([0, 1] * 10)
    p = np.linspace(0.01, 0.99, 20)
    captured = []

    def capture(fig, path, **kwargs):
        captured.append(fig)

    monkeypatch.setattr("ml_toolkit.plots.classification.save_figure", capture)
    try:
        curve = precision_recall(y, p, tmp_path / "pr.png", config=cfg)
        assert len(curve) == 20 and len(captured[0].axes[0].lines[0].get_xdata()) == 4
        groups = calibration(y, p, tmp_path / "cal.png", config=cfg)
        assert len(groups) == 2 and len(captured[1].axes[1].patches) == 4
    finally:
        for fig in captured:
            plt.close(fig)


def test_shap_sampling_honors_limit_and_seed():
    pytest.importorskip("shap")
    data = make_data()
    model = CatBoostRegressionModel(
        {"thread_count": 1, "iterations": 2, "depth": 2, "verbose": False, "allow_writing_files": False}
    ).fit(data)
    cfg = ShapConfig(max_samples=5, seed=23)
    first = calculate_shap_values(model.native_model, data.X, config=cfg)
    second = calculate_shap_values(model.native_model, data.X, config=cfg)
    other = calculate_shap_values(
        model.native_model, data.X, config=replace(cfg, seed=24)
    )
    assert first.values.shape == (5, 2)
    np.testing.assert_array_equal(first.data, second.data)
    assert not np.array_equal(first.data, other.data)
