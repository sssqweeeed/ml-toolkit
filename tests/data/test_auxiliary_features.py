"""Auxiliary column alignment, input isolation, and strict pool pairing."""

from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ml_toolkit.data import Pool, validate_training_data
from ml_toolkit.feature_levels.config import feature_definitions
from ml_toolkit.training import Split, TimeSeriesCV
from ml_toolkit.training.evaluation import _combine_pools


@pytest.fixture
def data():
    return Pool(
        pl.DataFrame({"x": np.arange(12, dtype=float)}),
        pl.Series(np.arange(12, dtype=float)),
        pl.Series([date(2025, 1, 1) + timedelta(days=i) for i in range(12)]),
        pl.Series([1.0] * 12),
        auxiliary_features=pl.DataFrame(
            {
                "segment": ["a", None, "b"] * 4,
                "amount": np.arange(12, dtype=float) * 100,
                "flag": [True, False] * 6,
            }
        ),
        numerical_features=("x",),
    )


def test_alignment_selection_and_buffer_sharing(data):
    for indices in ([3, 0, 3], range(2, 5)):
        selected = data.take(indices)
        assert_frame_equal(
            selected.auxiliary_features, data.auxiliary_features[list(indices)]
        )
    selected = data.take(range(2, 5))
    assert np.shares_memory(
        selected.auxiliary_features["amount"].to_numpy(),
        data.auxiliary_features["amount"].to_numpy(),
    )
    assert data.select_features(["x"]).auxiliary_features is data.auxiliary_features
    assert (
        data.with_target(np.zeros(len(data))).auxiliary_features
        is data.auxiliary_features
    )


def test_analysis_selection_is_named_ordered_and_does_not_mutate(data):
    frame = data.to_analysis_frame(["amount", "x", "segment"])
    assert frame.columns == ["amount", "x", "segment"]
    assert data.to_analysis_frame().columns == ["x", "segment", "amount", "flag"]
    frame[0, "amount"] = -1
    assert data.auxiliary_features["amount"][0] == 0
    with pytest.raises(ValueError, match="missing.*absent"):
        data.to_analysis_frame(["absent"])
    with pytest.raises(ValueError, match="unique"):
        data.to_analysis_frame(["x", "x"])
    assert replace(data, auxiliary_features=pl.DataFrame()).auxiliary_features is None
    assert replace(data, auxiliary_features=None).to_analysis_frame().equals(data.X)


def test_auxiliary_validation_and_model_isolation(data):
    with pytest.raises(ValueError, match="equal lengths"):
        replace(data, auxiliary_features=data.auxiliary_features.head(2))
    with pytest.raises(ValueError, match="overlap"):
        replace(data, auxiliary_features=data.X)
    with pytest.raises(ValueError, match="missing from X"):
        replace(data, numerical_features=("amount",))
    assert list(data.to_model_input().columns) == ["x"]
    assert data.to_catboost().get_feature_names() == ["x"]
    # Frames are not deeply immutable: revalidate at the training boundary.
    altered = replace(data, auxiliary_features=data.auxiliary_features.clone())
    altered.auxiliary_features.insert_column(0, data.X["x"])
    with pytest.raises(ValueError, match="overlap"):
        validate_training_data(altered)


@pytest.mark.parametrize("mismatch", ["absent", "dtype", "name", "order"])
def test_split_and_reporting_reject_incompatible_auxiliary_schemas(data, mismatch):
    train, test = data.take(range(8)), data.take(range(8, 12))
    frame = test.auxiliary_features
    replacement = {
        "absent": None,
        "dtype": frame.with_columns(pl.col("amount").cast(pl.Int64)),
        "name": frame.rename({"amount": "other"}),
        "order": frame.select("amount", "segment", "flag"),
    }[mismatch]
    test = replace(test, auxiliary_features=replacement)
    for operation in (lambda: Split(train, test), lambda: _combine_pools(train, test)):
        with pytest.raises(ValueError, match="auxiliary.*schemas"):
            operation()


def test_cv_and_reporting_preserve_auxiliary_columns(data):
    for part in TimeSeriesCV(data, n_splits=2).get_validation():
        combined = _combine_pools(part.train, part.heldout)
        ids = combined.X["x"].cast(pl.Int64)
        assert_frame_equal(combined.auxiliary_features, data.auxiliary_features[ids])
    inferred = {item.name: item.ftype for item in feature_definitions(data, None)}
    assert inferred == {
        "x": "numerical",
        "segment": "categorical",
        "amount": "numerical",
        "flag": "categorical",
    }
