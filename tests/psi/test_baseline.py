"""Persisted references reproduce row-based PSI without retaining source data."""

import json
from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ml_toolkit.feature_levels import FeatureLevelConfig, LevelTransformer
from ml_toolkit.psi import PSIBaseline, PSIAnalysis, PSIConfig


@pytest.fixture
def reference():
    return pl.DataFrame(
        {
            "x": [0.0, 1.0, 1.0, 2.0, 3.0, None, float("nan"), -999.0],
            "group": ["a", "b", "a", None, "a", "bad", "b", "a"],
            "constant": [4.0] * 8,
        }
    )


@pytest.fixture
def config():
    return PSIConfig(
        (
            FeatureLevelConfig(
                "x", "numerical", lower_bound=0, upper_bound=10, null_level=-999.0
            ),
            FeatureLevelConfig("group", "categorical", error_level="bad"),
            FeatureLevelConfig("constant", "numerical"),
        ),
        n_bins=4,
        epsilon=1e-5,
    )


def test_roundtrip_and_aggregate_scores(reference, config, tmp_path, monkeypatch):
    current = pl.DataFrame(
        {
            "x": [-1.0, 0.0, 1.0, 2.0, 10.0, 11.0, None],
            "group": ["new", "a", None, "b", "bad", "a", "a"],
            "constant": [1.0, 2.0, 3.0, 4.0, 5.0, None, 7.0],
            "extra": [42] * 7,
        }
    )
    expected = PSIAnalysis(config).fit(reference).compare(current)
    time = pl.Series(
        [datetime(2026, 10, 25, 2, 30, tzinfo=ZoneInfo("Europe/Berlin"))] * 8
    )
    baseline = PSIBaseline.from_reference(
        reference, config=config, time=time, stage="test"
    )
    baseline.save(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "levels.json",
        "manifest.json",
        "reference_counts.parquet",
    ]
    monkeypatch.setattr(
        LevelTransformer, "fit", lambda *a, **k: pytest.fail("must not refit")
    )
    loaded = PSIBaseline.load(tmp_path)
    result = loaded.compare(current)
    assert_frame_equal(result.psi, expected.psi)
    assert_frame_equal(result.level_distribution, expected.level_distribution)
    assert_frame_equal(
        loaded.transformer.transform(current), baseline.transformer.transform(current)
    )
    assert loaded.metadata == baseline.metadata
    assert result.time_distribution.is_empty()
    assert loaded.metadata["n_rows"] == 8
    assert not hasattr(loaded, "reference_levels")
    assert_frame_equal(
        baseline.compare(reference).psi.select("psi"), pl.DataFrame({"psi": [0.0] * 3})
    )


@pytest.mark.parametrize(
    "values",
    [
        [True, False, None],
        [1, 2, None],
        [1.25, 2.5, None],
        ["a", "b", None],
        [date(2024, 1, 1), date(2024, 1, 2), None],
        [
            datetime(2024, 1, 1, tzinfo=ZoneInfo("Europe/Moscow")),
            datetime(2024, 2, 1, tzinfo=ZoneInfo("Europe/Moscow")),
            None,
        ],
    ],
)
def test_typed_categories_roundtrip(values):
    data = pl.DataFrame({"category": values})
    transformer = LevelTransformer(
        (FeatureLevelConfig("category", "categorical", error_level=values[1]),)
    ).fit(data)
    restored = LevelTransformer.from_dict(
        json.loads(json.dumps(transformer.to_dict(), allow_nan=False))
    )
    assert restored.feature_levels == transformer.feature_levels
    assert_frame_equal(restored.transform(data), transformer.transform(data))


def test_exact_boundaries_infinity_and_special_values():
    data = pl.DataFrame({"x": [0.0, 1.0000000000001, 2.0000000000003, 3.0, None]})
    config = FeatureLevelConfig(
        "x",
        "numerical",
        lower_bound=-np.inf,
        upper_bound=np.inf,
        null_level=np.nan,
        error_level=-999.0,
    )
    transformer = LevelTransformer((config,), n_bins=3).fit(data)
    restored = LevelTransformer.from_dict(
        json.loads(json.dumps(transformer.to_dict(), allow_nan=False))
    )
    assert (
        restored.feature_levels[0].boundaries
        == transformer.feature_levels[0].boundaries
    )
    current = pl.DataFrame(
        {"x": [*transformer.feature_levels[0].boundaries, -999.0, np.nan, None]}
    )
    assert_frame_equal(restored.transform(current), transformer.transform(current))


@pytest.mark.parametrize(
    "series",
    [
        pl.Series("category", [1.1, 2.2, None], dtype=pl.Float32),
        pl.Series(
            "category",
            [datetime(2024, 1, 1), datetime(2024, 2, 1), None],
            dtype=pl.Datetime("ms"),
        ),
    ],
)
def test_category_labels_preserve_source_dtype_format(series, tmp_path):
    reference = pl.DataFrame(series)
    config = PSIConfig((FeatureLevelConfig("category", "categorical"),))
    baseline = PSIBaseline.from_reference(reference, config=config)
    baseline.save(tmp_path)
    loaded = PSIBaseline.load(tmp_path)
    assert loaded.transformer.feature_levels == baseline.transformer.feature_levels
    assert_frame_equal(
        loaded.compare(reference).psi,
        PSIAnalysis(config).fit(reference).compare(reference).psi,
    )


def test_changed_category_representation_cannot_silently_drop_counts():
    reference = pl.DataFrame({"x": pl.Series([1.1, 2.2], dtype=pl.Float32)})
    config = PSIConfig((FeatureLevelConfig("x", "categorical"),))
    baseline = PSIBaseline.from_reference(reference, config=config)
    with pytest.raises(ValueError, match="undefined levels"):
        baseline.compare(reference.cast(pl.Float64))


@pytest.mark.parametrize(
    "corruption",
    [
        "version",
        "epsilon",
        "rows",
        "levels",
        "negative",
        "missing",
        "duplicate",
        "null",
    ],
)
def test_reject_corrupted_baseline(reference, config, tmp_path, corruption):
    PSIBaseline.from_reference(reference, config=config).save(tmp_path)
    manifest = tmp_path / "manifest.json"
    if corruption in {"version", "epsilon", "rows"}:
        payload = json.loads(manifest.read_text())
        payload[{"rows": "n_rows"}.get(corruption, corruption)] = {
            "version": 2,
            "epsilon": 0,
            "rows": 99,
        }[corruption]
        manifest.write_text(json.dumps(payload))
    elif corruption == "levels":
        path = tmp_path / "levels.json"
        payload = json.loads(path.read_text())
        payload["features"][0]["boundaries"] = [{"type": "float", "value": "inf"}]
        path.write_text(json.dumps(payload))
    else:
        path = tmp_path / "reference_counts.parquet"
        counts = pl.read_parquet(path)
        if corruption == "negative":
            counts = counts.with_columns(pl.lit(-1, dtype=pl.Int64).alias("rows"))
        elif corruption == "null":
            counts = counts.with_columns(pl.lit(None, dtype=pl.Int64).alias("rows"))
        elif corruption == "missing":
            counts = counts.slice(1)
        else:
            counts = pl.concat([counts, counts.head(1)])
        counts.write_parquet(path)
    with pytest.raises(ValueError, match="Invalid PSI baseline"):
        PSIBaseline.load(tmp_path)


def test_invalid_inputs_and_incomplete_baseline(reference, config, tmp_path):
    with pytest.raises(RuntimeError, match="fitted"):
        LevelTransformer(config.features).to_dict()
    with pytest.raises(ValueError, match="empty"):
        PSIBaseline.from_reference(reference.head(0), config=config)
    baseline = PSIBaseline.from_reference(reference, config=config)
    with pytest.raises(ValueError, match="empty"):
        baseline.compare(reference.head(0))
    with pytest.raises(ValueError, match="missing"):
        baseline.compare(reference.drop("x"))
    with pytest.raises(FileNotFoundError, match="Incomplete PSI baseline"):
        PSIBaseline.load(tmp_path)


def test_coincident_interval_labels_keep_reference_counts_valid(tmp_path):
    reference = pl.DataFrame({"x": [1.0 + i * 1e-9 for i in range(100)]})
    current = reference.tail(25)
    config = PSIConfig((FeatureLevelConfig("x", "numerical"),))
    expected = PSIAnalysis(config).fit(reference).compare(current)
    baseline = PSIBaseline.from_reference(reference, config=config)
    baseline.save(tmp_path)
    restored = PSIBaseline.load(tmp_path)
    assert restored.reference_counts["rows"].sum() == len(reference)
    assert_frame_equal(restored.compare(current).psi, expected.psi)
    assert_frame_equal(
        restored.compare(current).level_distribution, expected.level_distribution
    )
