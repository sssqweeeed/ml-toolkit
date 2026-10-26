from dataclasses import FrozenInstanceError, replace
from datetime import date, timedelta

import catboost
import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

from ml_toolkit import Pool as PublicPool
from ml_toolkit.data import validate_training_categories, validate_training_data
from ml_toolkit.training import Pool as TrainingPool
from ml_toolkit.training.contracts import Pool as ContractPool
from examples.project_pool import ProjectPool as Pool


@pytest.fixture
def pool():
    return Pool(
        X=pl.DataFrame(
            {"x": [1.0, 2.0, 3.0], "category": [10, None, 20], "extra": [4.0, 5.0, 6.0]}
        ),
        y=pl.Series("target", [1.0, 2.0, 3.0]),
        time=pl.Series(
            "time", [date(2024, 1, 1) + timedelta(days=i) for i in range(3)]
        ),
        weight=pl.Series("weight", [1.0, 2.0, 3.0]),
        business_type=pl.Series("business_type", ["a", "b", "c"]),
        cat_features=("category",),
        numerical_features=("x",),
        object_id=pl.Series("customer_id", ["a", "b", "a"]),
    )


def test_public_pool_is_one_shared_container(pool):
    assert PublicPool is TrainingPool is ContractPool is Pool
    assert type(pool) is PublicPool


def test_project_pool_preserves_frozen_fields(pool):
    assert len(pool) == 3
    with pytest.raises(FrozenInstanceError):
        pool.cat_features = ()


@pytest.mark.parametrize(
    "field", ["X", "y", "time", "weight", "business_type", "object_id"]
)
def test_rejects_unaligned_fields(pool, field):
    with pytest.raises(ValueError, match="equal lengths"):
        replace(pool, **{field: getattr(pool, field).head(2)})


def test_rejects_empty_data(pool):
    with pytest.raises(ValueError, match="must not be empty"):
        pool.take([])


def test_rejects_non_temporal_time(pool):
    with pytest.raises(TypeError, match="Date or Datetime"):
        replace(pool, time=pl.Series("time", [1, 2, 3]))


@pytest.mark.parametrize("field", ["cat_features", "numerical_features"])
@pytest.mark.parametrize(
    "names, message", [(("missing",), "missing from X"), (("x", "x"), "duplicates")]
)
def test_rejects_invalid_feature_metadata(pool, field, names, message):
    with pytest.raises(ValueError, match=message):
        replace(pool, **{field: names})


def test_rejects_overlapping_feature_metadata(pool):
    with pytest.raises(ValueError, match="overlap"):
        replace(pool, numerical_features=("category",))


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("indices", [[2, 0, 2], np.array([2, 0, 2])])
def test_take_preserves_order_alignment_and_feature_metadata(pool, grouped, indices):
    if not grouped:
        pool = Pool(
            pool.X,
            pool.y,
            pool.time,
            pool.weight,
            cat_features=pool.cat_features,
            numerical_features=pool.numerical_features,
        )
    result = pool.take(indices)
    assert result is not pool
    assert result.X["x"].to_list() == [3.0, 1.0, 3.0]
    assert result.y.to_list() == [3.0, 1.0, 3.0]
    assert result.time.to_list() == [
        date(2024, 1, 3),
        date(2024, 1, 1),
        date(2024, 1, 3),
    ]
    assert result.weight.to_list() == [3.0, 1.0, 3.0]
    assert (None if result.object_id is None else result.object_id.to_list()) == (
        ["a", "a", "a"] if grouped else None
    )
    assert (
        None if result.business_type is None else result.business_type.to_list()
    ) == (["c", "a", "c"] if grouped else None)
    assert result.cat_features == pool.cat_features
    assert result.numerical_features == pool.numerical_features
    assert pool.X["x"].to_list() == [1.0, 2.0, 3.0]


@pytest.mark.parametrize("indices", [range(1, 3), [1, 2], np.arange(1, 3)])
def test_contiguous_take_shares_buffers_and_preserves_all_fields(pool, indices):
    result = pool.take(indices)
    assert_frame_equal(result.X, pool.X.slice(1, 2))
    assert np.shares_memory(result.X["x"].to_numpy(), pool.X["x"].to_numpy())
    for field in ("y", "time", "weight", "object_id", "business_type"):
        assert_series_equal(getattr(result, field), getattr(pool, field).slice(1, 2))
    for field in ("y", "time", "weight"):
        # Arrow exposes Date's actual int32 buffer; NumPy widens it to int64.
        original = getattr(pool, field).to_arrow().buffers()[1]
        selected = getattr(result, field).to_arrow().buffers()[1]
        assert selected.address == original.address
    result.X[0, "x"] = 99.0
    assert pool.X["x"].to_list() == [1.0, 2.0, 3.0]


@pytest.mark.parametrize(
    "indices", [[2, 1, 2], [0, 2], [-2, -1], [-1, 0], range(2, -1, -1)]
)
def test_take_fallback_matches_gather(pool, indices):
    index = np.asarray(indices, dtype=np.int64)
    result = pool.take(indices)
    assert_frame_equal(result.X, pool.X[index])
    for field in ("y", "time", "weight", "object_id", "business_type"):
        assert_series_equal(getattr(result, field), getattr(pool, field).gather(index))


@pytest.mark.parametrize("indices", [range(4), [1, 2, 3], [3]])
def test_contiguous_take_does_not_silently_clip_invalid_indices(pool, indices):
    with pytest.raises((IndexError, pl.exceptions.OutOfBoundsError)):
        pool.take(indices)


def test_select_features_preserves_rows_and_filters_both_feature_lists(pool):
    selected = pool.select_features(["category", "x"])
    assert selected.X.columns == ["category", "x"]
    assert selected.cat_features == ("category",)
    assert selected.numerical_features == ("x",)
    assert pool.select_features(["x"]).cat_features == ()
    assert pool.select_features(["category"]).numerical_features == ()
    for field in ("y", "time", "weight", "business_type", "object_id"):
        assert_series_equal(getattr(selected, field), getattr(pool, field))


@pytest.mark.parametrize(
    "names, message", [([], "At least one"), (["missing"], "missing from X")]
)
def test_select_features_rejects_invalid_names(pool, names, message):
    with pytest.raises(ValueError, match=message):
        pool.select_features(names)


@pytest.mark.parametrize(
    "target", [np.array([3, 2, 1]), pl.Series("replacement", [3, 2, 1])]
)
def test_with_target_preserves_metadata_and_names(pool, target):
    result = pool.with_target(target)
    assert result.y.to_list() == [3, 2, 1]
    assert result.y.name == (
        "replacement" if isinstance(target, pl.Series) else "target"
    )
    assert_frame_equal(result.X, pool.X)
    for field in ("time", "weight", "business_type", "object_id"):
        assert_series_equal(getattr(result, field), getattr(pool, field))
    assert result.cat_features == pool.cat_features
    assert result.numerical_features == pool.numerical_features
    assert pool.y.to_list() == [1.0, 2.0, 3.0]


@pytest.mark.parametrize(
    "target", [np.array([1, 2]), pl.Series([1, 2]), np.ones((3, 2))]
)
def test_rejects_invalid_replacement_target(pool, target):
    with pytest.raises(ValueError, match="one value per row"):
        pool.with_target(target)


@pytest.mark.parametrize("n_rows, expected", [(None, 3), (2, 2), (100, 3), (0, 0)])
def test_serving_input_preserves_order_and_source_data(pool, n_rows, expected):
    original = pool.X.clone()
    result = pool.to_model_input(n_rows)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == expected
    assert result.columns.to_list() == pool.X.columns
    assert result["x"].to_list() == [1.0, 2.0, 3.0][:expected]
    assert result["category"].to_list() == ["10", "unknown", "20"][:expected]
    pd.testing.assert_frame_equal(result, pool.to_model_input(n_rows))
    assert_frame_equal(pool.X, original)


def test_negative_serving_limit_is_rejected(pool):
    with pytest.raises(ValueError, match="nonnegative"):
        pool.to_model_input(-1)


@pytest.mark.parametrize("dtype", [pl.Categorical, pl.Enum(["a", "b"])])
@pytest.mark.parametrize("values", [["a", None, "b"], [None] * 3, ["b", "a", "b"]])
@pytest.mark.parametrize("chunked", [False, True])
def test_encoded_categories_stay_compact_and_match_string_input(
    pool, dtype, values, chunked
):
    frame = pool.X.with_columns(pl.Series("category", values, dtype=dtype))
    if chunked:
        frame = pl.concat([frame.head(1), frame.tail(2)], rechunk=False)
    data = replace(pool, X=frame)
    result = data.to_model_input()
    assert isinstance(result["category"].dtype, pd.CategoricalDtype)
    assert result["category"].tolist() == [
        "unknown" if v is None else v for v in values
    ]
    strings = data.to_model_input(categorical_as_strings=True)
    pd.testing.assert_frame_equal(result.astype({"category": str}), strings)
    assert data.X["category"].to_list() == values
    assert isinstance(data.to_model_input(0)["category"].dtype, pd.CategoricalDtype)


@pytest.mark.parametrize(
    "dtype", [pl.String, pl.Categorical, pl.Enum(["a", "unknown"])]
)
@pytest.mark.parametrize(
    "values", [["a", None, "a"], [None] * 3, ["a", "unknown", None]]
)
def test_reserved_category_checks_observed_values(pool, dtype, values):
    data = replace(
        pool, X=pool.X.with_columns(pl.Series("category", values, dtype=dtype))
    )
    if "unknown" in values:
        with pytest.raises(ValueError, match="category.*reserved"):
            validate_training_categories(data)
    else:
        validate_training_categories(data)


def test_reserved_category_check_accepts_enum_without_unknown(pool):
    data = replace(
        pool,
        X=pool.X.with_columns(
            pl.Series("category", ["a", None, "a"], dtype=pl.Enum(["a"]))
        ),
    )
    validate_training_categories(data)


@pytest.mark.parametrize("dtype", [pl.Float32, pl.Float64, pl.Int32, pl.Int64])
@pytest.mark.parametrize("n_rows", [None, 2])
def test_model_input_shares_numeric_buffers(pool, dtype, n_rows):
    pool = replace(pool, X=pool.X.with_columns(pl.col("x", "extra").cast(dtype)))
    result = pool.to_model_input(n_rows)
    source = pool.X if n_rows is None else pool.X.head(n_rows)
    expected = source.with_columns(
        pl.col("category").cast(pl.String).fill_null("unknown").cast(pl.Categorical)
    ).to_pandas()

    pd.testing.assert_frame_equal(result, expected)
    for name in ("x", "extra"):
        assert np.shares_memory(result[name].to_numpy(), pool.X[name].to_numpy())


@pytest.mark.parametrize("chunked", [False, True])
def test_model_input_preserves_nullable_numeric_columns(pool, chunked):
    frame = pool.X.with_columns(
        pl.Series("x", [1.0, None, float("nan")], dtype=pl.Float32)
    )
    if chunked:
        frame = pl.concat([frame.head(1), frame.tail(2)], rechunk=False)
    pool = replace(pool, X=frame)
    expected = frame.with_columns(
        pl.col("category").cast(pl.String).fill_null("unknown").cast(pl.Categorical)
    ).to_pandas()

    pd.testing.assert_frame_equal(pool.to_model_input(), expected)


@pytest.mark.parametrize(
    "dtype, baseline",
    [
        (pl.Date, None),
        (pl.Datetime("ms"), None),
        (pl.Datetime("us"), None),
        (pl.Datetime("ns", "UTC"), None),
        (pl.Date, 0.25),
        (pl.Date, np.array([0.1, 0.2, 0.3])),
    ],
)
def test_catboost_conversion(pool, dtype, baseline, monkeypatch):
    pool = replace(pool, time=pool.time.cast(dtype))
    original = pool.X.clone()
    kwargs = {}
    native_pool = catboost.Pool

    def capture(**values):
        kwargs.update(values)
        return native_pool(**values)

    monkeypatch.setattr(catboost, "Pool", capture)
    result = pool.to_catboost(baseline)
    assert isinstance(result, native_pool)
    assert result.num_row() == 3
    assert result.get_feature_names() == pool.X.columns
    assert result.get_cat_feature_indices() == [1]
    np.testing.assert_allclose(result.get_label(), pool.y.to_numpy())
    np.testing.assert_allclose(result.get_weight(), pool.weight.to_numpy())
    np.testing.assert_array_equal(
        kwargs["timestamp"], [1704067200000, 1704153600000, 1704240000000]
    )
    assert kwargs["data"]["category"].to_list() == ["10", "unknown", "20"]
    if baseline is None:
        assert kwargs["baseline"] is None
    else:
        expected = np.full(3, baseline) if np.isscalar(baseline) else baseline
        np.testing.assert_allclose(result.get_baseline().reshape(-1), expected)
    assert_frame_equal(pool.X, original)


def test_rejects_wrong_baseline_length(pool):
    with pytest.raises(ValueError, match="one value per row"):
        pool.to_catboost(np.array([0.0, 1.0]))


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("y", pl.Series("target", [1.0, np.nan, 3.0]), "Target values must be finite"),
        ("weight", pl.Series("weight", [1.0, -1.0, 1.0]), "Weights must be finite"),
        ("weight", pl.Series("weight", [0.0, 0.0, 0.0]), "Weights must be finite"),
        ("weight", pl.Series("weight", [1.0, np.inf, 1.0]), "Weights must be finite"),
        (
            "time",
            pl.Series("time", [date(2024, 1, 1), None, date(2024, 1, 3)]),
            "ascending timestamps",
        ),
    ],
)
def test_training_invariants_remain_in_training_validator(pool, field, value, message):
    data = replace(pool, **{field: value})
    with pytest.raises(ValueError, match=message):
        validate_training_data(data)


def test_reordered_pool_is_allowed_until_training(pool):
    data = pool.take([2, 1, 0])
    with pytest.raises(ValueError, match="ascending timestamps"):
        validate_training_data(data)


@pytest.mark.parametrize(
    "ids",
    [
        pl.Series("ids", [10, 20, 10], dtype=pl.Int64),
        pl.Series("ids", [10, 20, 10], dtype=pl.UInt64),
        pl.Series("ids", ["alpha", "beta", "alpha"]),
    ],
)
def test_object_ids_preserve_values_dtype_and_stay_out_of_model_inputs(pool, ids):
    data = replace(pool, object_id=ids)
    assert_series_equal(data.take([1, 0, 1]).object_id, ids.gather([1, 0, 1]))
    assert data.to_model_input().columns.to_list() == pool.X.columns
    assert data.to_catboost().get_feature_names() == pool.X.columns


@pytest.mark.parametrize(
    "ids, error, message",
    [
        (pl.Series([1, None, 3]), ValueError, "null"),
        (pl.Series([1.0, 2.0, 3.0]), ValueError, "string or integer"),
        (pl.Series([True, False, True]), ValueError, "string or integer"),
    ],
)
def test_pool_rejects_invalid_object_ids(pool, ids, error, message):
    with pytest.raises(error, match=message):
        replace(pool, object_id=ids)


def test_object_id_is_optional_and_preserves_positional_constructor(pool):
    data = Pool(
        pool.X,
        pool.y,
        pool.time,
        pool.weight,
        pool.business_type,
        pool.cat_features,
        pool.numerical_features,
    )
    assert data.object_id is None
