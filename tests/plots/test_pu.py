"""PU plots preserve weighted group semantics and review-capacity geometry."""

import numpy as np
import pytest
from ml_toolkit.plots import pu


def test_score_histograms_normalize_each_group_independently(monkeypatch, tmp_path):
    figures = []
    monkeypatch.setattr(
        pu, "save_figure", lambda fig, *args, **kwargs: figures.append(fig)
    )
    pu.score_distribution(
        [1, 1, 0, 0],
        [0.8, 0.2, 0.3, 0.9],
        tmp_path / "scores.png",
        weights=[3, 1, 2, 6],
        histogram_bins=2,
    )
    ax = figures[0].axes[0]
    assert [bar.get_height() for bar in ax.patches] == pytest.approx([0.25, 0.75] * 2)
    assert ax.get_legend_handles_labels()[1] == ["Known positive", "Unlabeled"]


def test_positive_capture_uses_weighted_percentages_and_keeps_ties(
    monkeypatch, tmp_path
):
    figures = []
    monkeypatch.setattr(
        pu, "save_figure", lambda fig, *args, **kwargs: figures.append(fig)
    )
    pu.positive_capture(
        [1, 0, 1, 0],
        [0.9, 0.8, 0.8, 0.1],
        tmp_path / "capture.png",
        weights=[1, 3, 3, 1],
    )
    ax = figures[0].axes[0]
    curve = ax.lines[0]
    np.testing.assert_allclose(curve.get_xdata(), [0, 0, 75, 100])
    np.testing.assert_allclose(curve.get_ydata(), [0, 25, 100, 100])
    legend = ax.get_legend()
    assert [text.get_text() for text in legend.get_texts()] == [
        "Model",
        "Random ranking",
    ]
    assert len(legend.get_lines()) == len(ax.lines) == 2
    for key, line in zip(legend.get_lines(), ax.lines):
        assert key.get_color() == line.get_color()
        assert key.get_linestyle() == line.get_linestyle()
