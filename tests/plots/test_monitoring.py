from datetime import date

import polars as pl

from ml_toolkit.plots.monitoring import time_feature_distribution


def test_time_numeric_distribution_summarizes_each_sample(tmp_path):
    reference = pl.DataFrame(
        {"date": [date(2025, 1, 1), date(2025, 1, 15)], "value": [1.0, 3.0]}
    )
    current = pl.DataFrame(
        {"date": [date(2025, 2, 1), date(2025, 2, 15)], "value": [2.0, 4.0]}
    )
    summary = time_feature_distribution(
        reference, current, "value", "date", tmp_path / "feature.png"
    )
    assert summary.height == 2
    assert summary.filter(pl.col("split") == "reference").get_column("mean").item() == 2
    assert (tmp_path / "feature.png").stat().st_size > 0
