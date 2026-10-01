"""Behavioral tests for the standalone Polars PSI package."""

import os
import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from ml_toolkit.psi import (
    ERROR_LEVEL,
    NULL_LEVEL,
    UNSEEN_LEVEL,
    LevelTransformer,
    PSIAnalysis,
    PSIConfig,
    PSIFeature,
    psi,
)

os.environ.setdefault("MPLBACKEND", "Agg")


class PSITests(unittest.TestCase):
    def test_score_uses_union_of_levels_and_smoothing(self):
        reference = pl.Series(["a", "a", "b"]).cast(pl.Categorical)
        self.assertEqual(psi(reference, reference), 0.0)
        current = pl.Series(["a", "c", "c"]).cast(pl.Categorical)
        self.assertGreater(psi(reference, current), 0.0)

    def test_numeric_and_categorical_level_rules(self):
        features = (
            PSIFeature("value", "numerical", lower_bound=0),
            PSIFeature("group", "categorical"),
        )
        transformer = LevelTransformer(features, n_bins=2).fit(
            pl.DataFrame({
                "value": [0.0, 10.0, 20.0, None],
                "group": ["a", "b", "a", None],
            })
        )
        result = transformer.transform(
            pl.DataFrame({
                "value": [-1.0, None, 30.0],
                "group": ["c", None, "a"],
            })
        )
        self.assertEqual(result.get_column("value").cast(pl.String)[0], ERROR_LEVEL)
        self.assertEqual(result.get_column("value").cast(pl.String)[1], NULL_LEVEL)
        self.assertEqual(result.get_column("group").cast(pl.String)[0], UNSEEN_LEVEL)
        self.assertEqual(result.get_column("group").cast(pl.String)[1], NULL_LEVEL)

    def test_analysis_preserves_feature_and_monthly_distributions(self):
        config = PSIConfig((
            PSIFeature("value", "numerical"),
            PSIFeature("group", "categorical"),
        ), n_bins=2)
        reference = pl.DataFrame({
            "value": [0.0, 10.0, 20.0, 30.0],
            "group": ["a", "b", "a", "b"],
            "date": [date(2025, 1, 1), date(2025, 1, 2), date(2025, 2, 1), date(2025, 2, 2)],
        })
        current = pl.DataFrame({
            "value": [5.0, 35.0, 50.0],
            "group": ["a", "c", "c"],
            "date": [date(2025, 3, 1)] * 3,
        })
        result = PSIAnalysis(config).fit(reference, time="date").compare(current, time="date")

        self.assertEqual(result.psi.get_column("feature").to_list(), ["value", "group"])
        self.assertTrue(np.isfinite(result.psi.get_column("psi").to_numpy()).all())
        shares = result.level_distribution.group_by("feature", "sample").agg(
            pl.col("share").sum().alias("total")
        )
        self.assertTrue(np.allclose(shares.get_column("total").to_numpy(), 1.0))
        monthly = result.monthly_distribution.group_by("feature", "sample", "month").agg(
            pl.col("share").sum().alias("total")
        )
        self.assertTrue(np.allclose(monthly.get_column("total").to_numpy(), 1.0))

        current_group = result.level_distribution.filter(
            (pl.col("feature") == "group") & (pl.col("sample") == "current")
        )
        unseen_share = current_group.filter(pl.col("level") == UNSEEN_LEVEL).item(0, "share")
        self.assertAlmostEqual(unseen_share, 2 / 3)

    def test_optional_time_and_validation(self):
        config = PSIConfig((PSIFeature("x", "numerical"),))
        reference = pl.DataFrame({"x": [1.0, 2.0]})
        current = pl.DataFrame({"x": [3.0, 4.0]})
        result = PSIAnalysis(config).fit(reference).compare(current)
        self.assertTrue(result.monthly_distribution.is_empty())
        self.assertEqual(
            result.monthly_distribution.columns,
            ["feature", "month", "sample", "level", "share"],
        )
        with self.assertRaisesRegex(ValueError, "Both samples"):
            PSIAnalysis(config).fit(reference, time=pl.Series([date(2025, 1, 1)] * 2)).compare(current)

    def test_plots_create_files(self):
        try:
            from ml_toolkit.plots.psi import feature_psi_plot, monthly_psi_level_distribution
        except ImportError:
            self.skipTest("Matplotlib plotting extra is not installed")
        config = PSIConfig((PSIFeature("x", "numerical"),))
        reference = pl.DataFrame({"x": [1.0, 2.0], "date": [date(2025, 1, 1)] * 2})
        current = pl.DataFrame({"x": [2.0, 3.0], "date": [date(2025, 2, 1)] * 2})
        result = PSIAnalysis(config).fit(reference, time="date").compare(current, time="date")
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "summary.png"
            monthly = Path(directory) / "monthly.png"
            feature_psi_plot(result.psi, summary)
            monthly_psi_level_distribution(result.monthly_distribution, "x", monthly)
            self.assertGreater(summary.stat().st_size, 0)
            self.assertGreater(monthly.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
