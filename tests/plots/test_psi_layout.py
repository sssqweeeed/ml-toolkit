"""Calendar geometry and sample separation in the shared PSI panel."""

from datetime import datetime
from zoneinfo import ZoneInfo

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import polars as pl
import pytest

from ml_toolkit.plots import psi


@pytest.mark.parametrize(
    "period,reference,current",
    [
        ("minute", datetime(2025, 1, 1), datetime(2025, 1, 1, 0, 3)),
        ("hour", datetime(2025, 1, 1), datetime(2025, 1, 1, 3)),
        ("day", datetime(2025, 1, 1), datetime(2025, 1, 3)),
        ("week", datetime(2025, 1, 6), datetime(2025, 1, 20)),
        ("month", datetime(2025, 1, 1), datetime(2025, 4, 1)),
        ("quarter", datetime(2025, 1, 1), datetime(2025, 10, 1)),
        ("year", datetime(2025, 1, 1), datetime(2027, 1, 1)),
    ],
)
def test_one_calendar_panel_marks_boundary_and_keeps_gaps(
    period, reference, current, monkeypatch, tmp_path
):
    zone = ZoneInfo("Europe/Berlin")
    reference, current = reference.replace(tzinfo=zone), current.replace(tzinfo=zone)
    distribution = pl.DataFrame(
        {
            "feature": ["x"] * 4,
            "period_start": [reference, reference, current, current],
            "sample": ["reference", "reference", "current", "current"],
            "level": ["low", "high", "low", "high"],
            "share": [0.25, 0.75, 0.6, 0.4],
        }
    )
    figures = []
    monkeypatch.setattr(psi, "save_figure", lambda fig, *a, **k: figures.append(fig))
    try:
        psi.time_psi_level_distribution(
            distribution, "x", tmp_path / "psi.png", period=period
        )
        assert len(figures[0].axes) == 1
        ax = figures[0].axes[0]
        assert len(ax.patches) == 4
        reference_bar, current_bar = ax.patches[:2]
        assert reference_bar.get_x() == pytest.approx(mdates.date2num(reference))
        assert current_bar.get_x() == pytest.approx(mdates.date2num(current))
        assert reference_bar.get_x() + reference_bar.get_width() < current_bar.get_x()
        assert reference_bar.get_height() == 0.25
        assert current_bar.get_height() == 0.6
        assert list(ax.lines[0].get_xdata()) == pytest.approx(
            [mdates.date2num(current)] * 2
        )
        assert {text.get_text() for text in ax.texts} == {"REFERENCE", "CURRENT"}
        assert [text.get_text() for text in ax.get_legend().get_texts()] == [
            "low",
            "high",
        ]
    finally:
        for fig in figures:
            plt.close(fig)


@pytest.mark.parametrize(
    "levels,shares",
    [
        pytest.param(
            ["[0, 1)", "__NULL__", "__ERROR__", "__UNSEEN__"],
            [0.4, 0.3, 0.2, 0.1],
            id="mixed-levels",
        ),
        pytest.param(
            ["__NULL__", "__ERROR__", "__UNSEEN__"],
            [0.5, 0.3, 0.2],
            id="special-levels-only",
        ),
        pytest.param(["__NULL__"], [1.0], id="null-only"),
        pytest.param(["regular", "_custom"], [0.6, 0.4], id="custom-underscore"),
        pytest.param(
            ["a", "b", "c", "d", "__NULL__", "__ERROR__", "__UNSEEN__", "_custom"],
            [0.1, 0.2, 0.1, 0.2, 0.0, 0.1, 0.1, 0.2],
            id="multiple-rows-with-zero-share",
        ),
    ],
)
def test_legend_keeps_all_level_labels_and_matches_bar_colors(
    levels, shares, monkeypatch, tmp_path
):
    distribution = pl.DataFrame(
        {
            "feature": ["x"] * len(levels),
            "period_start": [datetime(2025, 1, 1)] * len(levels),
            "sample": ["reference"] * len(levels),
            "level": levels,
            "share": shares,
        }
    )
    figures = []
    monkeypatch.setattr(psi, "save_figure", lambda fig, *a, **k: figures.append(fig))
    try:
        psi.time_psi_level_distribution(distribution, "x", tmp_path / "psi.png")
        ax = figures[0].axes[0]
        legend = ax.get_legend()
        assert [text.get_text() for text in legend.get_texts()] == levels
        assert len(legend.get_patches()) == len(levels)
        assert [bar.get_height() for bar in ax.patches] == pytest.approx(shares)
        for index, (key, bar) in enumerate(zip(legend.get_patches(), ax.patches)):
            assert key.get_facecolor() == pytest.approx(bar.get_facecolor())
            assert bar.get_facecolor() == pytest.approx(plt.get_cmap("tab20")(index))
    finally:
        for fig in figures:
            plt.close(fig)


def test_shared_period_shows_both_samples_without_changing_table(monkeypatch, tmp_path):
    distribution = pl.DataFrame(
        {
            "feature": ["x", "x"],
            "period_start": [datetime(2025, 1, 1)] * 2,
            "sample": ["current", "reference"],
            "level": ["low", "low"],
            "share": [0.3, 0.7],
        }
    )
    before = distribution.clone()
    figures = []
    monkeypatch.setattr(psi, "save_figure", lambda fig, *a, **k: figures.append(fig))
    try:
        psi.time_psi_level_distribution(distribution, "x", tmp_path / "psi.png")
        assert distribution.equals(before)
        assert len(figures[0].axes) == 1
        ax = figures[0].axes[0]
        reference, current = ax.patches
        assert reference.get_height() == 0.7
        assert current.get_height() == 0.3
        assert reference.get_x() == pytest.approx(mdates.date2num(datetime(2025, 1, 1)))
        assert reference.get_x() + reference.get_width() == pytest.approx(
            current.get_x()
        )
        assert current.get_x() + current.get_width() == pytest.approx(
            mdates.date2num(datetime(2025, 2, 1))
        )
        assert ax.lines[0].get_xdata()[0] == current.get_x()
    finally:
        for fig in figures:
            plt.close(fig)
