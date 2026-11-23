"""Interval labels stay unique and infinities are valid numerical values."""

import json
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
        shares = result.level_distribution.group_by("sample").agg(pl.col("share").sum())
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


def _state(
    transformer: LevelTransformer,
) -> tuple[dict[str, object], dict[str, object]]:
    state = transformer.to_dict()
    return state, cast(list[dict[str, object]], state["features"])[0]


def _legacy_labels(transformer: LevelTransformer, suffix: list[str]) -> list[str]:
    assert transformer.feature_levels is not None
    levels = transformer.feature_levels[0]
    lower = -np.inf if levels.config.lower_bound is None else levels.config.lower_bound
    upper = np.inf if levels.config.upper_bound is None else levels.config.upper_bound
    edges = (lower, *levels.boundaries, upper)
    labels = [
        f"[{'-inf' if a == -np.inf else f'{a:g}'}, "
        f"{'inf' if b == np.inf else f'{b:g}'}"
        f"{']' if b == upper and b != np.inf else ')'}"
        for a, b in zip(edges, edges[1:])
    ]
    return [*labels, *suffix]


def test_new_labels_equal_legacy_labels_for_ordinary_bounded_data():
    config = FeatureLevelConfig("x", "numerical", lower_bound=0, upper_bound=10)
    data = pl.DataFrame({"x": [0.5, 1.25, 2.0, 3.5, 10.0, 7.0, 0.1, 4.4]})
    fitted = LevelTransformer([config], n_bins=4).fit(data)
    assert fitted.feature_levels is not None
    assert list(fitted.feature_levels[0].levels) == _legacy_labels(
        fitted, ["__NULL__", "__ERROR__"]
    )


def test_legacy_labels_with_bounds_and_error_suffix_load():
    config = FeatureLevelConfig(
        "x", "numerical", lower_bound=0, upper_bound=10, error_level=-1.0
    )
    data = pl.DataFrame({"x": [0.5, 1.25, 2.0, 3.5, 10.0, 7.0, 0.1, 4.4]})
    fitted = LevelTransformer([config], n_bins=4).fit(data)
    assert fitted.feature_levels is not None
    legacy = _legacy_labels(fitted, ["__NULL__", "__ERROR__"])
    state, feature = _state(fitted)
    feature["levels"] = legacy
    loaded = LevelTransformer.from_dict(state)
    assert loaded.feature_levels is not None
    assert loaded.feature_levels[0].levels == fitted.feature_levels[0].levels


def test_labels_matching_neither_form_are_rejected():
    data = pl.DataFrame({"x": [0.5, 1.25, 2.0, 3.5, 10.0, 7.0, 0.1, 4.4]})
    state, feature = _state(LevelTransformer([X], n_bins=4).fit(data))
    feature["levels"] = ["[-inf, 1)", "[1, 2)", "[2, 3)", "[3, inf)", "__NULL__"]
    with pytest.raises(ValueError, match="do not match"):
        LevelTransformer.from_dict(state)


def test_infinity_with_upper_bound_only_and_explicit_infinite_bounds():
    train = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    batch = pl.DataFrame({"x": [-np.inf, np.inf]})

    upper_only = FeatureLevelConfig("x", "numerical", upper_bound=10)
    fitted = LevelTransformer([upper_only], n_bins=2).fit(train)
    assert fitted.feature_levels is not None
    assert _labels(fitted, batch) == [fitted.feature_levels[0].levels[0], ERROR_LEVEL]

    explicit = FeatureLevelConfig(
        "x", "numerical", lower_bound=-np.inf, upper_bound=np.inf
    )
    fitted = LevelTransformer([explicit], n_bins=2).fit(train)
    assert fitted.feature_levels is not None
    levels = fitted.feature_levels[0].levels
    assert _labels(fitted, batch) == [levels[0], levels[1]]


def test_baseline_compares_infinite_current_values():
    reference = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    config = PSIConfig((X,), n_bins=3)
    baseline = PSIBaseline.from_reference(reference, config=config)
    current = pl.DataFrame({"x": [-np.inf, np.inf, 2.0, 5.0]})
    result = baseline.compare(current)
    expected = PSIAnalysis(config).fit(reference).compare(current)
    assert result.psi.equals(expected.psi)
    assert np.isfinite(result.psi.get_column("psi").to_numpy()).all()


def _write_legacy_baseline(tmp_path, reference: pl.DataFrame, merge: bool) -> None:
    baseline = PSIBaseline.from_reference(reference, config=PSIConfig((X,), n_bins=10))
    baseline.save(tmp_path)
    legacy = _legacy_labels(baseline.transformer, ["__NULL__"])
    state = json.loads((tmp_path / "levels.json").read_text())
    state["features"][0]["levels"] = legacy
    (tmp_path / "levels.json").write_text(json.dumps(state))
    if merge:
        counts = baseline.reference_counts
        merged = (
            counts.with_columns(pl.Series("level", legacy))
            .group_by("feature", "level", maintain_order=True)
            .agg(pl.col("rows").sum())
        )
        merged.select(*counts.columns).write_parquet(
            tmp_path / "reference_counts.parquet"
        )


def test_legacy_baseline_with_duplicate_labels_is_rejected(tmp_path):
    _write_legacy_baseline(tmp_path, _large_values(0), merge=True)
    with pytest.raises(ValueError, match="ambiguous interval labels"):
        PSIBaseline.load(tmp_path)


def test_legacy_baseline_without_duplicates_loads_and_compares(tmp_path):
    reference = pl.DataFrame({"x": [0.5, 1.25, 2.0, 3.5, 10.0, 7.0, 0.1, 4.4]})
    _write_legacy_baseline(tmp_path, reference, merge=False)
    loaded = PSIBaseline.load(tmp_path)
    expected = PSIAnalysis(PSIConfig((X,), n_bins=10)).fit(reference).compare(reference)
    assert loaded.compare(reference).psi.equals(expected.psi)
