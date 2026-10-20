import numpy as np
import polars as pl
import pytest

from ml_toolkit.config import FeatureLevelConfig
from ml_toolkit.feature_levels import LevelTransformer
from ml_toolkit.plots.regression import observed_fitted_by_level
from ml_toolkit.plots.utils import PlotError


def test_weighted_comparison_keeps_level_order_and_gaps(tmp_path):
    order = ("[0, 2)", "[2, 10)", "[10, inf)", "__NULL__", "__ERROR__", "__UNSEEN__")
    summary = observed_fitted_by_level(
        pl.Series([order[2], order[0], order[0], order[3], order[4]]),
        [10, 1, 5, 7, 99],
        [9, 2, 6, 8, 99],
        pl.Series([order[1], order[2], order[5]]),
        [3, 12, 4],
        [4, 11, 5],
        tmp_path / "comparison.png",
        level_order=order,
        reference_weight=[2, 1, 3, 1, 0],
        current_weight=[1, 2, 1],
    )
    reference = summary.loc[summary["sample"] == "reference"].set_index("level")
    current = summary.loc[summary["sample"] == "current"].set_index("level")
    assert list(reference.index) == [order[0], order[1], order[2], order[3], order[5]]
    assert reference.loc[order[0], "observed"] == 4.0
    assert reference.loc[order[0], "fitted"] == 5.0
    assert reference.loc[order[0], "volume"] == 4.0
    assert np.isnan(reference.loc[order[1], "observed"])
    assert reference.loc[order[1], "volume"] == 0
    assert np.isnan(current.loc[order[0], "fitted"])
    assert (tmp_path / "comparison.png").stat().st_size > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reference_observed": [1]},
        {"current_fitted": [1, np.inf]},
        {"reference_weight": [-1, 2]},
        {"current_weight": [0, 0]},
        {"level_order": ("a", "a")},
        {"current_levels": pl.Series(["a", "unknown"])},
        {"reference_levels": pl.Series(["a", None])},
    ],
)
def test_observed_fitted_rejects_invalid_inputs(tmp_path, kwargs):
    inputs = dict(
        reference_levels=pl.Series(["a", "b"]),
        reference_observed=[1, 2],
        reference_fitted=[1, 2],
        current_levels=pl.Series(["a", "b"]),
        current_observed=[1, 2],
        current_fitted=[1, 2],
        level_order=("a", "b"),
        save_fig=tmp_path / "invalid.png",
    )
    inputs.update(kwargs)
    with pytest.raises(PlotError):
        observed_fitted_by_level(**inputs)


def test_special_levels_and_constant_features_are_shared_with_psi(tmp_path):
    from ml_toolkit.psi import compare_psi

    reference = pl.DataFrame(
        {
            "value": [0, 0, 10, 10, None, -1, -999],
            "group": ["a", "b", "a", "b", None, "missing", "error"],
            "constant": [5] * 7,
        }
    )
    current = pl.DataFrame(
        {
            "value": [-2, 0, 10, 21, None, -1, -999],
            "group": ["a", "new", "b", "new", None, "missing", "error"],
            "constant": [5] * 7,
        }
    )
    transformer = LevelTransformer(
        (
            FeatureLevelConfig(
                "value",
                "numerical",
                lower_bound=0,
                upper_bound=20,
                null_level=-1,
                error_level=-999,
            ),
            FeatureLevelConfig(
                "group", "categorical", null_level="missing", error_level="error"
            ),
            FeatureLevelConfig("constant", "numerical"),
        ),
        n_bins=10,
    ).fit(reference)
    ref_levels, cur_levels = transformer.transform(reference), transformer.transform(
        current
    )
    assert transformer.feature_levels is not None
    assert len(transformer.feature_levels[0].boundaries) < 9
    assert transformer.feature_levels[2].boundaries == ()
    assert cur_levels["value"].cast(pl.String).to_list()[3:] == [
        "__ERROR__",
        "__NULL__",
        "__NULL__",
        "__ERROR__",
    ]
    assert cur_levels["group"].cast(pl.String).to_list() == [
        "a",
        "__UNSEEN__",
        "b",
        "__UNSEEN__",
        "__NULL__",
        "__NULL__",
        "__ERROR__",
    ]
    result = compare_psi(ref_levels, cur_levels, transformer.feature_levels)
    for feature in transformer.feature_levels:
        table = observed_fitted_by_level(
            ref_levels[feature.config.name],
            np.arange(7),
            np.arange(7),
            cur_levels[feature.config.name],
            np.arange(7),
            np.arange(7),
            tmp_path / f"{feature.config.name}.png",
            level_order=feature.levels,
        )
        distribution = result.level_distribution.filter(
            pl.col("feature") == feature.config.name
        )
        for row in table.itertuples():
            count = distribution.filter(
                (pl.col("sample") == row.sample) & (pl.col("level") == row.level)
            )["rows"].item()
            assert row.volume == count
