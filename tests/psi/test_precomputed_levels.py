"""Public PSI comparison rejects tables not produced by its fitted definitions."""

import polars as pl
import pytest

from ml_toolkit.feature_levels import FeatureLevelConfig, LevelTransformer
from ml_toolkit.psi import compare_psi


@pytest.mark.parametrize("sample", ["reference", "current"])
@pytest.mark.parametrize(
    "invalid,message,error",
    [
        (pl.DataFrame({"other": ["a"]}), "missing", ValueError),
        (pl.DataFrame({"group": ["a"]}), "Categorical", TypeError),
        (
            pl.DataFrame({"group": ["new"]}).cast(pl.Categorical),
            "undefined",
            ValueError,
        ),
        (
            pl.DataFrame({"group": [None]}, schema={"group": pl.Categorical}),
            "undefined",
            ValueError,
        ),
        (pl.DataFrame(schema={"group": pl.Categorical}), "empty", ValueError),
    ],
)
def test_compare_validates_both_precomputed_level_tables(
    sample, invalid, message, error
):
    frame = pl.DataFrame({"group": ["a"]})
    transformer = LevelTransformer((FeatureLevelConfig("group", "categorical"),)).fit(
        frame
    )
    valid = transformer.transform(frame)
    assert transformer.feature_levels is not None
    reference, current = (invalid, valid) if sample == "reference" else (valid, invalid)
    with pytest.raises(error, match=message):
        compare_psi(reference, current, transformer.feature_levels)
