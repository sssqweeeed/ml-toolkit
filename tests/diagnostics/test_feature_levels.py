import polars as pl
import pytest

from ml_toolkit.feature_levels import (
    ERROR_LEVEL,
    NULL_LEVEL,
    UNSEEN_LEVEL,
    FeatureLevelSpec,
    LevelTransformer,
)


def test_numeric_and_categorical_level_rules():
    features = (
        FeatureLevelSpec("value", "numerical", lower_bound=0),
        FeatureLevelSpec("group", "categorical"),
    )
    transformer = LevelTransformer(features, n_bins=2).fit(
        pl.DataFrame({"value": [0.0, 10.0, 20.0, None], "group": ["a", "b", "a", None]})
    )
    result = transformer.transform(
        pl.DataFrame({"value": [-1.0, None, 30.0], "group": ["c", None, "a"]})
    )
    assert result.get_column("value").cast(pl.String)[0] == ERROR_LEVEL
    assert result.get_column("value").cast(pl.String)[1] == NULL_LEVEL
    assert result.get_column("group").cast(pl.String)[0] == UNSEEN_LEVEL
    assert result.get_column("group").cast(pl.String)[1] == NULL_LEVEL


def test_numeric_boundaries_are_left_inclusive_and_keep_passthrough_rows():
    transformer = LevelTransformer(
        (FeatureLevelSpec("x", "numerical", lower_bound=0, upper_bound=10),),
        n_bins=2,
    ).fit(pl.DataFrame({"x": [0.0, 10.0]}))
    frame = pl.DataFrame(
        {"x": [-1.0, 0.0, 4.0, 5.0, 10.0, 11.0, None, float("nan")], "id": range(8)}
    )
    result = transformer.transform(frame)
    assert result["x"].cast(pl.String).to_list() == [
        ERROR_LEVEL,
        "[0, 5)",
        "[0, 5)",
        "[5, 10]",
        "[5, 10]",
        ERROR_LEVEL,
        NULL_LEVEL,
        NULL_LEVEL,
    ]
    assert result["x"].dtype == pl.Categorical
    assert result["id"].equals(frame["id"])
    assert frame.schema["x"] == pl.Float64


def test_transform_requires_fit_and_all_configured_features():
    transformer = LevelTransformer((FeatureLevelSpec("x", "numerical"),))
    frame = pl.DataFrame({"x": [1.0]})
    with pytest.raises(RuntimeError, match="fitted"):
        transformer.transform(frame)
    transformer.fit(frame)
    with pytest.raises(ValueError, match="missing.*x"):
        transformer.transform(frame.rename({"x": "other"}))


@pytest.mark.parametrize("reserved", [NULL_LEVEL, ERROR_LEVEL, UNSEEN_LEVEL])
def test_categorical_reference_rejects_reserved_diagnostic_labels(reserved):
    transformer = LevelTransformer((FeatureLevelSpec("group", "categorical"),))
    with pytest.raises(ValueError, match="reserved"):
        transformer.fit(pl.DataFrame({"group": ["regular", reserved]}))
