from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.metrics import f1_score, precision_score, recall_score

from examples.project_pool import ProjectPool
from ml_toolkit.models import CatBoostRegressionModel, CatBoostClassificationModel
from ml_toolkit.models._catboost import prepare_frame
from ml_toolkit.plots.classification import binary_curves


def pool():
    return ProjectPool(
        X=pl.DataFrame({"a": range(12), "b": range(12), "c": range(12)}),
        y=pl.Series([0.0, 1.0] * 6),
        time=pl.Series([date(2025, 1, 1) + timedelta(days=i) for i in range(12)]),
        weight=pl.Series([1.0] * 12),
        numerical_features=("a", "b", "c"),
    )


@pytest.mark.parametrize("kind", [CatBoostRegressionModel, CatBoostClassificationModel])
def test_reserved_categories_checked_only_at_fit(kind):
    data = pool()
    data = replace(
        data,
        X=data.X.with_columns(pl.Series("cat", [None, "a"] * 6)),
        cat_features=("cat",),
    )
    params = {
        "iterations": 2,
        "thread_count": 1,
        "verbose": False,
        "allow_writing_files": False,
    }
    model = kind(params).fit(data)
    np.testing.assert_allclose(
        model.predict_pool(data), model.predict_frame(data.to_model_input())
    )
    invalid = replace(data, X=data.X.with_columns(pl.lit("unknown").alias("cat")))
    with pytest.raises(ValueError, match="cat.*reserved"):
        kind(params).fit(invalid)
    with pytest.raises(ValueError, match="cat.*reserved"):
        kind(params).fit(data, invalid)
    assert np.isfinite(model.predict_frame(invalid.to_model_input())).all()


def test_pandas_preparation_reuses_numeric_buffers_without_mutating_categories():
    data = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0], "cat": [10, None, 20], "b": [4.0, 5.0, 6.0]},
        index=[7, 8, 9],
    )
    original = data.copy(deep=True)
    result = prepare_frame(data, ["b", "cat", "a"], ["cat"])
    assert result.columns.to_list() == ["b", "cat", "a"]
    pd.testing.assert_index_equal(result.index, data.index)
    for name in ("a", "b"):
        assert np.shares_memory(result[name].to_numpy(), data[name].to_numpy())
    assert result["cat"].to_list() == ["10.0", "unknown", "20.0"]
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize(
    "values", [["b", "a", "b"], [1, 2, 1], [1, "1", None], [None] * 3]
)
def test_pandas_encoded_categories_match_previous_string_values(values):
    data = pd.DataFrame(
        {"cat": pd.Categorical(values), "a": [1.0, 2.0, 3.0]}, index=[7, 8, 9]
    )
    original = data.copy(deep=True)
    expected = data["cat"].astype("string").fillna("unknown").astype(str)
    result = prepare_frame(data, ["a", "cat"], ["cat"])
    assert isinstance(result["cat"].dtype, pd.CategoricalDtype)
    pd.testing.assert_series_equal(result["cat"].astype(str), expected)
    pd.testing.assert_frame_equal(data, original)
    if values == ["b", "a", "b"]:
        assert np.shares_memory(data["cat"].array.codes, result["cat"].array.codes)


@pytest.mark.parametrize("kind", [CatBoostRegressionModel, CatBoostClassificationModel])
def test_encoded_predictions_match_strings_with_different_dictionary_order(
    kind, monkeypatch
):
    data = replace(
        pool(),
        X=pool().X.with_columns(
            pl.Series("cat", ["a", "b", None] * 4, dtype=pl.Enum(["a", "b"]))
        ),
        cat_features=("cat",),
    )
    params = {
        "iterations": 4,
        "thread_count": 1,
        "verbose": False,
        "allow_writing_files": False,
    }
    model = kind(params).fit(data)
    strings = data.to_model_input(categorical_as_strings=True)
    convert = ProjectPool.to_model_input
    with monkeypatch.context() as patch:
        patch.setattr(
            ProjectPool,
            "to_model_input",
            lambda self, n_rows=None: convert(
                self, n_rows, categorical_as_strings=True
            ),
        )
        reference = kind(params).fit(data)
    np.testing.assert_array_equal(
        model.predict_pool(data), reference.predict_pool(data)
    )
    encoded = strings.copy()
    encoded["cat"] = pd.Categorical(strings["cat"], categories=["unknown", "b", "a"])
    np.testing.assert_array_equal(
        model.predict_frame(encoded), model.predict_frame(strings)
    )
    np.testing.assert_array_equal(
        model.predict_pool(data), model.predict_frame(strings)
    )


@pytest.mark.parametrize("dtype", [pl.Float32, pl.Float64, pl.Int32, pl.Int64])
def test_polars_preparation_shares_numeric_buffers(dtype):
    data = pl.DataFrame(
        {"a": [1, 2, 3], "cat": ["north", None, "south"], "b": [4, 5, 6]}
    ).with_columns(pl.col("a", "b").cast(dtype), pl.col("cat").cast(pl.Categorical))
    original = data.clone()
    result = prepare_frame(data, ["b", "cat", "a"], ["cat"])
    expected = (
        data.select("b", "cat", "a")
        .with_columns(
            pl.col("cat").cast(pl.String).fill_null("unknown").cast(pl.Categorical)
        )
        .to_pandas()
    )

    pd.testing.assert_frame_equal(result, expected)
    for name in ("a", "b"):
        assert np.shares_memory(result[name].to_numpy(), data[name].to_numpy())
    assert data.equals(original)


@pytest.mark.parametrize("chunked", [False, True])
def test_polars_preparation_preserves_nullable_numeric_columns(chunked):
    data = pl.DataFrame(
        {
            "a": [1.0, None, float("nan")],
            "cat": ["north", None, "south"],
            "b": [4.0, 5.0, 6.0],
        }
    ).with_columns(pl.col("a", "b").cast(pl.Float32))
    if chunked:
        data = pl.concat([data.head(1), data.tail(2)], rechunk=False)
    expected = (
        data.select("b", "cat", "a")
        .with_columns(
            pl.col("cat").cast(pl.String).fill_null("unknown").cast(pl.Categorical)
        )
        .to_pandas()
    )

    pd.testing.assert_frame_equal(
        prepare_frame(data, ["b", "cat", "a"], ["cat"]), expected
    )


@pytest.mark.parametrize("kind", [CatBoostRegressionModel, CatBoostClassificationModel])
def test_pool_prediction_prepares_only_model_columns_once(kind, monkeypatch):
    data = replace(
        pool(),
        X=pool().X.with_columns(pl.Series("cat", [None, "north"] * 6)),
        cat_features=("cat",),
    )
    model = kind(
        {
            "iterations": 2,
            "thread_count": 1,
            "verbose": False,
            "allow_writing_files": False,
        }
    ).fit(data)
    expected = model.predict_frame(data.to_model_input())
    data = replace(data, X=data.X.with_columns(pl.lit("ignored").alias("extra")))
    conversions = []
    to_pandas = pl.DataFrame.to_pandas

    def capture(frame, *args, **kwargs):
        conversions.append(frame.columns)
        return to_pandas(frame, *args, **kwargs)

    monkeypatch.setattr(pl.DataFrame, "to_pandas", capture)
    np.testing.assert_array_equal(model.predict_pool(data), expected)
    assert conversions == [["a", "b", "c", "cat"]]
    if isinstance(model, CatBoostClassificationModel):
        conversions.clear()
        np.testing.assert_array_equal(
            model.predict_labels_pool(data),
            (expected >= model.threshold).astype(np.int64),
        )
        assert conversions == [["a", "b", "c", "cat"]]


def test_weighted_curves_match_metrics_and_ignore_zero_rows():
    y = np.array([1.0, 0.0, 1.0, 0.0])
    p = np.array([0.9, 0.7, 0.4, 1.0])
    w = np.array([5.0, 20.0, 1.0, 0.0])
    curve = binary_curves(y, p, weight=w)
    for row in curve.itertuples():
        labels = p >= row.threshold
        assert row.f1 == pytest.approx(f1_score(y, labels, sample_weight=w))
        assert row.precision == pytest.approx(
            precision_score(y, labels, sample_weight=w)
        )
        assert row.recall == pytest.approx(recall_score(y, labels, sample_weight=w))
    threshold, score = CatBoostClassificationModel._select_threshold(y, p, w)
    assert threshold == curve.iloc[curve.f1.argmax()].threshold
    assert score == curve.f1.max()
    assert threshold != CatBoostClassificationModel._select_threshold(y[:3], p[:3])[0]
    pd.testing.assert_frame_equal(curve, binary_curves(y[:3], p[:3], weight=w[:3]))
    pd.testing.assert_frame_equal(
        binary_curves(y, p), binary_curves(y, p, weight=np.ones(4))
    )
    assert CatBoostClassificationModel._select_threshold(
        np.array([1.0, 0.0]), np.array([0.8, 0.2]), np.array([0.0, 1.0])
    ) == (0.5, 0.0)


def test_weighted_threshold_prefers_larger_threshold_on_exact_ties():
    y, p, w = (
        np.array([1.0, 0.0, 1.0]),
        np.array([0.9, 0.8, 0.7]),
        np.array([1.0, 2.0, 1.0]),
    )
    threshold, f1 = CatBoostClassificationModel._select_threshold(y, p, w)
    assert threshold == 0.9
    assert f1 == pytest.approx(2 / 3)
    tied = binary_curves(y, np.full(3, 0.6), weight=w)
    assert tied.threshold.to_list() == [0.6]
    assert tied.f1.to_list() == pytest.approx([2 / 3])
