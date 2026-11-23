"""Interval labels stay unique and infinities are valid numerical values."""

from typing import cast

import numpy as np
import polars as pl
import pytest

from ml_toolkit.feature_levels import ERROR_LEVEL, FeatureLevelConfig, LevelTransformer
from ml_toolkit.psi import PSIAnalysis, PSIBaseline, PSIConfig

X = FeatureLevelConfig("x", "numerical")


def _large_values(seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame({"x": 1234567 + rng.uniform(0, 9, 500)})


def _labels(transformer: LevelTransformer, data: pl.DataFrame) -> list[str]:
    return transformer.transform(data).get_column("x").cast(pl.String).to_list()


def test_ordinary_labels_are_unchanged():
    data = pl.DataFrame({"x": [0.5, 1.25, 2.0, 3.5, 10.0, 7.0, 0.1, 4.4]})
    fitted = LevelTransformer([X], n_bins=4).fit(data)
    assert fitted.feature_levels is not None
    assert fitted.feature_levels[0].levels == (
        "[-inf, 1.0625)",
        "[1.0625, 2.75)",
        "[2.75, 5.05)",
        "[5.05, inf)",
        "__NULL__",
    )


def test_close_boundaries_get_unique_labels_and_unit_shares():
    reference, current = _large_values(0), _large_values(1)
    config = PSIConfig((X,), n_bins=10)
    fitted = LevelTransformer([X], n_bins=10).fit(reference)
    assert fitted.feature_levels is not None
    levels = fitted.feature_levels[0].levels
    assert len(set(levels)) == len(levels) == 11

    baseline = PSIBaseline.from_reference(reference, config=config)
    for result in (
        PSIAnalysis(config).fit(reference).compare(current),
        baseline.compare(current),
    ):
        shares = result.level_distribution.group_by("sample").agg(
            pl.col("share").sum()
        )
        assert shares.get_column("share").to_list() == pytest.approx([1.0, 1.0])
        assert result.level_distribution.get_column("level").n_unique() == 11


def test_legacy_state_with_duplicate_labels_loads_with_unique_labels():
    reference = _large_values(0)
    fitted = LevelTransformer([X], n_bins=10).fit(reference)
    assert fitted.feature_levels is not None
    state = fitted.to_dict()
    feature = cast(list[dict[str, object]], state["features"])[0]
    edges = (-np.inf, *fitted.feature_levels[0].boundaries, np.inf)
    legacy = [
        f"[{'-inf' if a == -np.inf else f'{a:g}'}, "
        f"{'inf' if b == np.inf else f'{b:g}'})"
        for a, b in zip(edges, edges[1:])
    ]
    assert len(set(legacy)) < len(legacy)
    feature["levels"] = [*legacy, "__NULL__"]

    loaded = LevelTransformer.from_dict(state)
    assert loaded.feature_levels is not None
    assert loaded.feature_levels[0].levels == fitted.feature_levels[0].levels
    assert _labels(loaded, reference) == _labels(fitted, reference)


def test_infinities_are_valid_in_fit_and_transform():
    reference = pl.DataFrame({"x": [-np.inf, 1.0, 2.0, 3.0, 4.0, np.inf]})
    fitted = LevelTransformer([X], n_bins=2).fit(reference)
    assert fitted.feature_levels is not None
    assert fitted.feature_levels[0].boundaries == (2.5,)
    levels = fitted.feature_levels[0].levels

    out = _labels(fitted, reference)
    assert out[0] == levels[0]
    assert out[-1] == levels[1]
    assert _labels(fitted, pl.DataFrame({"x": [-np.inf, np.inf]})) == [
        levels[0],
        levels[1],
    ]


def test_infinities_outside_bounds_become_error_level():
    bounded = FeatureLevelConfig("x", "numerical", lower_bound=0, upper_bound=10)
    reference = pl.DataFrame({"x": [-np.inf, 1.0, 2.0, 3.0, np.inf]})
    fitted = LevelTransformer([bounded], n_bins=2).fit(reference)
    out = _labels(fitted, reference)
    assert out[0] == out[-1] == ERROR_LEVEL
