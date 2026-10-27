"""The custom ranker guide preserves the model and aligned observation data."""

from datetime import date

import numpy as np
import polars as pl
import pytest

from examples.custom_ranker import PermutationMAERanker
from ml_toolkit.data import Pool

pytestmark = pytest.mark.documentation


class SignalModel:
    def __init__(self):
        self.inputs = []

    def predict_pool(self, data):
        self.inputs.append(data)
        return data.X["signal"].to_numpy()


def sample():
    return Pool(
        X=pl.DataFrame({"signal": [0.0, 1.0, 4.0, 9.0], "group": ["a", "b", "a", "b"]}),
        y=pl.Series("target", [0.0, 2.0, 4.0, 9.0]),
        time=pl.Series("time", [date(2025, 1, day) for day in range(1, 5)]),
        weight=pl.Series("weight", [1.0, 2.0, 3.0, 4.0]),
        auxiliary_features=(pl.Series("business", ["x", "y", "x", "y"])).to_frame(),
        numerical_features=("signal",),
        cat_features=("group",),
        object_id=pl.Series("id", [1, 1, 2, 2]),
    )


def test_permutation_ranker_uses_weighted_error_increase():
    data = sample()
    model = SignalModel()
    ranker = PermutationMAERanker(seed=42)
    result = ranker(model, data)
    # Seed 42 reverses these four signal values. Weighted MAE rises from .2
    # to 5.8; permuting the unused category leaves predictions unchanged.
    assert result == pytest.approx({"signal": 5.6, "group": 0.0})
    assert ranker(model, data) == result
    assert tuple(result) == tuple(data.X.columns)


def test_permutation_ranker_preserves_original_rows_and_metadata():
    data = sample()
    original = data.X.clone()
    model = SignalModel()
    PermutationMAERanker()(model, data)
    assert len(model.inputs) == 1 + data.X.width
    assert data.X.equals(original)
    for perturbed in model.inputs:
        assert perturbed.y is data.y
        assert perturbed.time is data.time
        assert perturbed.weight is data.weight
        assert perturbed.object_id is data.object_id
        assert perturbed.auxiliary_features is data.auxiliary_features
        assert perturbed.cat_features == data.cat_features
        assert perturbed.numerical_features == data.numerical_features
        assert perturbed.X.schema == data.X.schema
    assert model.inputs[1].X["group"].equals(data.X["group"])
    np.testing.assert_array_equal(model.inputs[2].X["signal"], data.X["signal"])
