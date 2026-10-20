from datetime import date

import polars as pl

from ml_toolkit.feature_levels import FeatureLevelConfig
from ml_toolkit.plots.psi import feature_psi_plot, time_psi_level_distribution
from ml_toolkit.psi import PSIAnalysis, PSIConfig


def test_psi_plots_export_nonempty_files(tmp_path):
    config = PSIConfig((FeatureLevelConfig("x", "numerical"),))
    reference = pl.DataFrame({"x": [1.0, 2.0], "date": [date(2025, 1, 1)] * 2})
    current = pl.DataFrame({"x": [2.0, 3.0], "date": [date(2025, 2, 1)] * 2})
    result = (
        PSIAnalysis(config).fit(reference, time="date").compare(current, time="date")
    )
    summary, monthly = tmp_path / "summary.png", tmp_path / "monthly.png"
    feature_psi_plot(result.psi, summary)
    time_psi_level_distribution(result.time_distribution, "x", monthly)
    assert summary.stat().st_size > 0
    assert monthly.stat().st_size > 0
