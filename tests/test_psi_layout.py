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
        assert ax.get_legend_handles_labels()[1] == ["low", "high"]
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
