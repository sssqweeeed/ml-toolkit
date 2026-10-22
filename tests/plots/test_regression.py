from datetime import date

from ml_toolkit.plots.regression import time_observed_fitted, observed_fitted_by_level
import polars as pl


def test_regression_weighted_aggregations(tmp_path):
    summary = observed_fitted_by_level(
        pl.Series(["a", "a"]),
        [1, 3],
        [2, 2],
        pl.Series(["b"]),
        [4],
        [5],
        tmp_path / "feature.png",
        level_order=("a", "b"),
        reference_weight=[1, 3],
        current_weight=[1],
    )
    assert (
        summary.loc[
            (summary.level == "a") & (summary["sample"] == "reference"), "observed"
        ].iloc[0]
        == 2.5
    )
    monthly = time_observed_fitted(
        [date(2025, 1, 1), date(2025, 1, 20)],
        [1, 3],
        [2, 2],
        [date(2025, 2, 1)],
        [4],
        [5],
        tmp_path / "monthly.png",
        reference_weight=[1, 3],
        reference_group=["a", "a"],
        current_group=["a"],
    )
    assert monthly.loc[monthly.split == "reference", "observed"].iloc[0] == 2.5
    assert (tmp_path / "feature.png").is_file() and (tmp_path / "monthly.png").is_file()
