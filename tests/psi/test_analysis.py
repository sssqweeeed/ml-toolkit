from datetime import date

import numpy as np
import polars as pl
import pytest

from ml_toolkit.feature_levels import (
    UNSEEN_LEVEL,
    FeatureLevelSpec,
)
from ml_toolkit.psi import PSIAnalysis, PSIConfig


def test_analysis_preserves_feature_and_time_distributions():
    config = PSIConfig(
        (
            FeatureLevelSpec("value", "numerical"),
            FeatureLevelSpec("group", "categorical"),
        ),
        n_bins=2,
    )
    reference = pl.DataFrame(
        {
            "value": [0.0, 10.0, 20.0, 30.0],
            "group": ["a", "b", "a", "b"],
            "date": [
                date(2025, 1, 1),
                date(2025, 1, 2),
                date(2025, 2, 1),
                date(2025, 2, 2),
            ],
        }
    )
    current = pl.DataFrame(
        {
            "value": [5.0, 35.0, 50.0],
            "group": ["a", "c", "c"],
            "date": [date(2025, 3, 1)] * 3,
        }
    )
    result = (
        PSIAnalysis(config).fit(reference, time="date").compare(current, time="date")
    )
    assert result.psi.get_column("feature").to_list() == ["value", "group"]
    assert np.isfinite(result.psi.get_column("psi").to_numpy()).all()
    shares = result.level_distribution.group_by("feature", "sample").agg(
        pl.col("share").sum().alias("total")
    )
    assert np.allclose(shares.get_column("total").to_numpy(), 1.0)
    monthly = result.time_distribution.group_by(
        "feature", "sample", "period_start"
    ).agg(pl.col("share").sum().alias("total"))
    assert np.allclose(monthly.get_column("total").to_numpy(), 1.0)
    current_group = result.level_distribution.filter(
        (pl.col("feature") == "group") & (pl.col("sample") == "current")
    )
    unseen_share = current_group.filter(pl.col("level") == UNSEEN_LEVEL).item(
        0, "share"
    )
    assert unseen_share == pytest.approx(2 / 3)


def test_optional_time_and_validation():
    config = PSIConfig((FeatureLevelSpec("x", "numerical"),))
    reference = pl.DataFrame({"x": [1.0, 2.0]})
    current = pl.DataFrame({"x": [3.0, 4.0]})
    result = PSIAnalysis(config).fit(reference).compare(current)
    assert result.time_distribution.is_empty()
    assert result.time_distribution.columns == [
        "feature",
        "period_start",
        "sample",
        "level",
        "share",
    ]
    with pytest.raises(ValueError, match="Both samples"):
        PSIAnalysis(config).fit(
            reference, time=pl.Series([date(2025, 1, 1)] * 2)
        ).compare(current)


def test_compare_requires_fitted_reference():
    analysis = PSIAnalysis(PSIConfig((FeatureLevelSpec("x", "numerical"),)))
    with pytest.raises(RuntimeError, match="fitted"):
        analysis.compare(pl.DataFrame({"x": [1.0]}))


@pytest.mark.parametrize("stage", ["fit", "compare"])
@pytest.mark.parametrize(
    "time,error,message",
    [
        (pl.Series([date(2025, 1, 1)]), ValueError, "align"),
        (pl.Series([date(2025, 1, 1), None]), ValueError, "null"),
        (pl.Series([1, 2]), TypeError, "Date or Datetime"),
    ],
)
def test_analysis_validates_time_at_each_public_boundary(stage, time, error, message):
    analysis = PSIAnalysis(PSIConfig((FeatureLevelSpec("x", "numerical"),)))
    frame = pl.DataFrame({"x": [1.0, 2.0]})
    if stage == "compare":
        analysis.fit(frame)
    with pytest.raises(error, match=message):
        getattr(analysis, stage)(frame, time=time)


def test_repeated_comparisons_preserve_reference_levels_until_refit():
    config = PSIConfig((FeatureLevelSpec("group", "categorical"),))
    original = pl.DataFrame({"group": ["a", "a"]})
    new = pl.DataFrame({"group": ["b", "b"]})
    analysis = PSIAnalysis(config).fit(original)
    first = analysis.compare(new)
    repeated = analysis.compare(new)
    assert first.psi.equals(repeated.psi)
    assert analysis.compare(original).psi["psi"].item() == 0
    unseen = first.level_distribution.filter(
        (pl.col("sample") == "current") & (pl.col("level") == UNSEEN_LEVEL)
    )
    assert unseen["share"].item() == 1
    assert analysis.fit(new).compare(new).psi["psi"].item() == 0
