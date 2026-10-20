"""Gallery examples retain real source observations and honest holdouts."""

from datetime import datetime

import numpy as np
import polars as pl
import pytest

pytestmark = pytest.mark.documentation
from polars.testing import assert_frame_equal
from sklearn.datasets import load_digits

from examples.gallery_data import (
    BIKE_FEATURES,
    DEFAULT_DATA_HOME,
    bike_cache_available,
    load_bike_pool,
    load_gallery_data,
)


@pytest.fixture(scope="module")
def gallery():
    if not bike_cache_available():
        pytest.skip("The real Bike Sharing gallery requires a locally cached hour.csv")
    return load_gallery_data(download_if_missing=False)


def test_missing_cache_never_downloads_when_disabled(tmp_path, monkeypatch):
    def unexpected_download(*args, **kwargs):
        pytest.fail("Gallery tests must not download datasets")

    monkeypatch.setattr("examples.gallery_data.urlretrieve", unexpected_download)
    assert not bike_cache_available(tmp_path)
    with pytest.raises(FileNotFoundError, match="Bike Sharing data is not cached"):
        load_gallery_data(data_home=tmp_path, download_if_missing=False)
    with pytest.raises(FileNotFoundError, match="Bike Sharing data is not cached"):
        load_bike_pool(data_home=tmp_path, download_if_missing=False)
    assert list(tmp_path.iterdir()) == []


def test_training_pool_retains_original_bike_rows_and_timestamps():
    if not bike_cache_available():
        pytest.skip("The full frequency example needs cached UCI Bike Sharing data")
    data = load_bike_pool(download_if_missing=False)
    source = pl.read_csv(DEFAULT_DATA_HOME / "hour.csv")
    assert len(data) == len(source) == 17379
    assert_frame_equal(data.X, source.select(*BIKE_FEATURES).rename(BIKE_FEATURES))
    np.testing.assert_array_equal(data.y, source["cnt"])
    expected_time = source.select(
        pl.col("dteday").str.to_datetime("%Y-%m-%d") + pl.duration(hours=pl.col("hr"))
    ).to_series()
    np.testing.assert_array_equal(data.time, expected_time)
    np.testing.assert_array_equal(data.weight, np.ones(len(source)))
    assert data.numerical_features == tuple(data.X.columns)


def test_digits_have_a_stratified_source_holdout_and_real_model_probabilities(gallery):
    digits, _ = gallery
    features, labels = load_digits(return_X_y=True)
    assert len(digits.train_indices) == 1078
    assert len(digits.test_indices) == len(digits.observed) == 719
    assert np.intersect1d(digits.train_indices, digits.test_indices).size == 0
    np.testing.assert_array_equal(
        np.sort(np.concatenate((digits.train_indices, digits.test_indices))),
        np.arange(1797),
    )
    np.testing.assert_array_equal(digits.digit_labels, labels[digits.test_indices])
    np.testing.assert_array_equal(digits.observed, labels[digits.test_indices] >= 5)
    assert np.bincount(digits.observed.astype(np.int64)).min() >= 350
    train_rate = (labels[digits.train_indices] >= 5).mean()
    assert abs(train_rate - digits.observed.mean()) < 0.005
    assert np.unique(digits.probabilities).size > 500
    assert np.isfinite(digits.probabilities).all()
    assert ((digits.probabilities >= 0) & (digits.probabilities <= 1)).all()
    np.testing.assert_allclose(
        digits.probabilities,
        digits.native_model.predict_proba(features[digits.test_indices])[:, 1],
    )
    np.testing.assert_array_equal(digits.weight, np.ones(719))


def test_bike_features_and_targets_match_source_rows_without_target_leakage(gallery):
    _, bikes = gallery
    source = pl.read_csv(DEFAULT_DATA_HOME / "hour.csv")
    assert len(source) == 17379
    assert len(bikes.train_indices) + len(bikes.test_indices) == len(source)
    assert np.intersect1d(bikes.train_indices, bikes.test_indices).size == 0
    np.testing.assert_array_equal(
        np.sort(np.concatenate((bikes.train_indices, bikes.test_indices))),
        np.arange(len(source)),
    )
    assert bikes.features.columns == list(BIKE_FEATURES.values())
    assert {"cnt", "casual", "registered", "instant", "source_row", "yr"}.isdisjoint(
        bikes.features.columns
    )
    for cohort in (bikes.reference, bikes.current):
        original = source[cohort.source_indices]
        assert_frame_equal(
            cohort.features, original.select(*BIKE_FEATURES).rename(BIKE_FEATURES)
        )
        np.testing.assert_array_equal(cohort.observed, original["cnt"].to_numpy())
        expected_time = original.select(
            pl.col("dteday").str.to_datetime("%Y-%m-%d")
            + pl.duration(hours=pl.col("hr"))
        ).to_series()
        np.testing.assert_array_equal(cohort.time.to_numpy(), expected_time.to_numpy())
        np.testing.assert_array_equal(cohort.weight, np.ones(len(cohort.features)))


def test_bike_training_precedes_both_real_periods_and_months_are_well_populated(gallery):
    _, bikes = gallery
    source = pl.read_csv(DEFAULT_DATA_HOME / "hour.csv")
    training_dates = source[bikes.train_indices]["dteday"].str.to_date()
    assert training_dates.max() < datetime(2011, 7, 1).date()
    assert bikes.reference.time.min() >= datetime(2011, 7, 1)
    assert bikes.reference.time.max() < datetime(2012, 1, 1)
    assert bikes.current.time.min() >= datetime(2012, 1, 1)
    assert bikes.current.time.max() < datetime(2013, 1, 1)
    assert len(bikes.reference.features) >= 4000
    assert len(bikes.current.features) >= 8000
    np.testing.assert_array_equal(
        np.concatenate((bikes.reference.source_indices, bikes.current.source_indices)),
        bikes.test_indices,
    )
    for cohort, expected_months in ((bikes.reference, 6), (bikes.current, 12)):
        months = cohort.time.dt.truncate("1mo")
        assert months.n_unique() == expected_months
        assert months.value_counts()["count"].min() >= 500


def test_bike_predictions_and_shap_rows_come_from_the_actual_fitted_model(gallery):
    _, bikes = gallery
    assert bikes.features.shape == (len(bikes.test_indices), 11)
    assert bikes.shap_features.shape == (600, 11)
    assert bikes.shap_features.join(
        bikes.features, on=bikes.features.columns, how="anti"
    ).is_empty()
    np.testing.assert_allclose(
        bikes.prediction, bikes.native_model.predict(bikes.features.to_pandas())
    )
    np.testing.assert_allclose(
        np.concatenate((bikes.reference.prediction, bikes.current.prediction)),
        bikes.prediction,
    )
    assert np.isfinite(bikes.prediction).all()
    assert list(bikes.importance) == bikes.features.columns
    assert np.isfinite(list(bikes.importance.values())).all()
    assert sum(bikes.importance.values()) > 0
