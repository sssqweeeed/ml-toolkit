"""Real held-out data and fitted models for the documentation gallery.

Classification uses scikit-learn's bundled handwritten digits: digits 0--4 are
negative and 5--9 are positive. A fixed, stratified 60/40 split keeps model
training separate from every displayed classification observation.

Regression uses the official UCI Bike Sharing hourly records. The model fits
January--June 2011, the reference sample is July--December 2011, and the current
sample is January--December 2012. Dates combine the actual dteday and hr fields.
Features are renamed source columns; targets are actual rental counts. Casual
and registered counts are excluded because their sum reveals the target.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve
from zipfile import ZipFile

import catboost
import numpy as np
import polars as pl
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from ml_toolkit.types import FloatArray, IntArray

from ml_toolkit.data import Pool


BIKE_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
DEFAULT_DATA_HOME = Path(__file__).resolve().parents[1] / ".cache/uci-bike-sharing"
BIKE_FEATURES = {
    "temp": "temperature",
    "atemp": "feels_like_temperature",
    "hum": "humidity",
    "windspeed": "wind_speed",
    "hr": "hour",
    "workingday": "working_day",
    "weekday": "weekday",
    "holiday": "holiday",
    "season": "season",
    "weathersit": "weather",
    "mnth": "month",
}


@dataclass(frozen=True)
class DigitsGallery:
    """Store the real supervised digits holdout and its fitted classifier.

    Attributes
    ----------
    observed : FloatArray
        Binary labels for the held-out digit images; 1 means digits 5--9.
    probabilities : FloatArray
        Positive-class probabilities from a classifier fitted on training only.
    weight : FloatArray
        Unit weights: every held-out image contributes equally.
    digit_labels : IntArray
        Original digit labels, aligned with the binary labels.
    train_indices, test_indices : IntArray
        Disjoint row positions in the original 1,797-image source dataset.
    native_model : catboost.CatBoostClassifier
        Fitted classifier; no test rows were used as an evaluation set.
    """

    observed: FloatArray
    probabilities: FloatArray
    weight: FloatArray
    digit_labels: IntArray
    train_indices: IntArray
    test_indices: IntArray
    native_model: catboost.CatBoostClassifier


@dataclass(frozen=True)
class BikeCohort:
    """Store aligned held-out hourly observations for one real calendar period.

    Attributes
    ----------
    features : polars.DataFrame
        Eleven weather and calendar columns, renamed from the source records.
    observed, prediction, weight : FloatArray
        Actual rental counts, predictions, and unit weights, one value per hour.
    time : polars.Series
        Actual source dates and hours as naive Datetime values.
    source_indices : IntArray
        Row positions in the original hour.csv source file.
    """

    features: pl.DataFrame
    observed: FloatArray
    prediction: FloatArray
    weight: FloatArray
    time: pl.Series
    source_indices: IntArray


@dataclass(frozen=True)
class BikeGallery:
    """Store genuine future-period bike rentals and their training-only model.

    Attributes
    ----------
    features : polars.DataFrame
        All held-out observations, July 2011--December 2012, in time order.
    observed, prediction, weight : FloatArray
        Aligned actual counts, predictions, and unit hourly weights.
    train_indices, test_indices : IntArray
        Disjoint row positions in the original hourly source file.
    reference, current : BikeCohort
        July--December 2011 and January--December 2012 observations.
    native_model : catboost.CatBoostRegressor
        Model fitted only on January--June 2011 observations.
    shap_features : polars.DataFrame
        Fixed-seed sample of 600 actual held-out rows, without targets or dates.
    importance : dict[str, float]
        Native importance for each of the eleven model features.
    """

    features: pl.DataFrame
    observed: FloatArray
    prediction: FloatArray
    weight: FloatArray
    train_indices: IntArray
    test_indices: IntArray
    reference: BikeCohort
    current: BikeCohort
    native_model: catboost.CatBoostRegressor
    shap_features: pl.DataFrame
    importance: dict[str, float]


def bike_cache_available(data_home: Path | None = None) -> bool:
    """Check whether the official hourly CSV is cached without downloading it.

    Parameters
    ----------
    data_home : pathlib.Path or None, default None
        Cache directory; None uses the repository's .cache/uci-bike-sharing.

    Returns
    -------
    bool
        Whether hour.csv exists in the selected cache directory.
    """
    directory = DEFAULT_DATA_HOME if data_home is None else data_home
    return (directory / "hour.csv").is_file()


def _bike_records(data_home: Path, download_if_missing: bool) -> pl.DataFrame:
    csv_path = data_home / "hour.csv"
    if not csv_path.is_file():
        if not download_if_missing:
            raise FileNotFoundError(
                f"Bike Sharing data is not cached at {csv_path}; "
                "run the gallery generator with downloads enabled first"
            )
        data_home.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="bike-download-", dir=data_home) as temporary:
            archive_path = Path(temporary) / "bike-sharing.zip"
            urlretrieve(BIKE_URL, archive_path)
            with ZipFile(archive_path) as archive:
                extracted = Path(temporary) / "hour.csv"
                extracted.write_bytes(archive.read("hour.csv"))
                extracted.replace(csv_path)
    return (
        pl.read_csv(csv_path)
        .with_row_index("source_row")
        .with_columns(
            (
                pl.col("dteday").str.to_datetime("%Y-%m-%d")
                + pl.duration(hours=pl.col("hr"))
            ).alias("timestamp")
        )
        .sort("timestamp")
    )


def _digits_gallery() -> DigitsGallery:
    features, digits = load_digits(return_X_y=True)
    observed = (digits >= 5).astype(np.float64)
    train, test = train_test_split(
        np.arange(len(digits), dtype=np.int64),
        test_size=0.4,
        random_state=42,
        stratify=observed,
    )
    classifier = catboost.CatBoostClassifier(
        iterations=200,
        depth=5,
        learning_rate=0.06,
        loss_function="Logloss",
        random_seed=42,
        thread_count=2,
        verbose=False,
        allow_writing_files=False,
    ).fit(features[train], observed[train])
    return DigitsGallery(
        observed=observed[test],
        probabilities=classifier.predict_proba(features[test])[:, 1],
        weight=np.ones(len(test), dtype=np.float64),
        digit_labels=digits[test],
        train_indices=train,
        test_indices=test,
        native_model=classifier,
    )


def load_bike_pool(
    *, data_home: Path | None = None, download_if_missing: bool = True
) -> Pool:
    """Load the original hourly records for a chronological training example.

    Parameters
    ----------
    data_home : pathlib.Path or None, default None
        Cache containing hour.csv; None uses the gallery's repository cache.
    download_if_missing : bool, default True
        Download the official UCI archive when absent. False fails without
        downloading. The loader does not fit any model or sample any rows.

    Returns
    -------
    Pool
        All 17,379 recorded hours in timestamp order, with eleven weather and
        calendar features, rental counts as targets, and unit hourly weights.
        Counts revealing the target (casual and registered) are excluded.
    """
    directory = DEFAULT_DATA_HOME if data_home is None else data_home
    records = _bike_records(directory, download_if_missing)
    features = records.select(*BIKE_FEATURES).rename(BIKE_FEATURES)
    return Pool(
        X=features,
        y=records["cnt"].cast(pl.Float64),
        time=records["timestamp"],
        weight=pl.Series("weight", np.ones(len(records), dtype=np.float64)),
        cat_features=(),
        numerical_features=tuple(features.columns),
    )


def _bike_gallery(records: pl.DataFrame) -> BikeGallery:
    train = records.filter(pl.col("timestamp") < datetime(2011, 7, 1))
    holdout = records.filter(pl.col("timestamp") >= datetime(2011, 7, 1))
    training_features = train.select(*BIKE_FEATURES).rename(BIKE_FEATURES)
    model = catboost.CatBoostRegressor(
        iterations=250,
        depth=6,
        learning_rate=0.06,
        loss_function="RMSE",
        random_seed=42,
        thread_count=2,
        verbose=False,
        allow_writing_files=False,
    ).fit(training_features.to_pandas(), train["cnt"].to_numpy())
    heldout_features = holdout.select(*BIKE_FEATURES).rename(BIKE_FEATURES)
    predictions = model.predict(heldout_features.to_pandas())
    holdout = holdout.with_columns(pl.Series("prediction", predictions))
    reference_records = holdout.filter(pl.col("timestamp") < datetime(2012, 1, 1))
    current_records = holdout.filter(pl.col("timestamp") >= datetime(2012, 1, 1))
    reference = BikeCohort(
        features=reference_records.select(*BIKE_FEATURES).rename(BIKE_FEATURES),
        observed=reference_records["cnt"].cast(pl.Float64).to_numpy(),
        prediction=reference_records["prediction"].to_numpy(),
        weight=np.ones(len(reference_records), dtype=np.float64),
        time=reference_records["timestamp"],
        source_indices=reference_records["source_row"].to_numpy().astype(np.int64),
    )
    current = BikeCohort(
        features=current_records.select(*BIKE_FEATURES).rename(BIKE_FEATURES),
        observed=current_records["cnt"].cast(pl.Float64).to_numpy(),
        prediction=current_records["prediction"].to_numpy(),
        weight=np.ones(len(current_records), dtype=np.float64),
        time=current_records["timestamp"],
        source_indices=current_records["source_row"].to_numpy().astype(np.int64),
    )
    return BikeGallery(
        features=heldout_features,
        observed=holdout["cnt"].cast(pl.Float64).to_numpy(),
        prediction=predictions,
        weight=np.ones(len(holdout), dtype=np.float64),
        train_indices=train["source_row"].to_numpy().astype(np.int64),
        test_indices=holdout["source_row"].to_numpy().astype(np.int64),
        reference=reference,
        current=current,
        native_model=model,
        shap_features=heldout_features.sample(n=600, seed=42),
        importance={
            name: float(value)
            for name, value in zip(
                heldout_features.columns, model.get_feature_importance(), strict=True
            )
        },
    )


def load_gallery_data(
    *,
    data_home: Path | None = None,
    download_if_missing: bool = True,
) -> tuple[DigitsGallery, BikeGallery]:
    """Train deterministic models and return their real held-out gallery data.

    Parameters
    ----------
    data_home : pathlib.Path or None, default None
        Directory containing UCI hour.csv. None uses .cache/uci-bike-sharing
        in the repository. Digits ships with scikit-learn and needs no download.
    download_if_missing : bool, default True
        Allow downloading the official UCI archive when hour.csv is absent.
        False raises FileNotFoundError without network requests or cache writes.

    Returns
    -------
    digits : DigitsGallery
        719 supervised holdout images and their classifier probabilities.
    bikes : BikeGallery
        Real future hourly rentals, calendar cohorts, model, and SHAP sample.

    Notes
    -----
    Models fit their training rows only, without test evaluation sets. Unit
    weights make each observation equally influential. Source targets and
    covariates are used unchanged; no observations, dates, or shifts are
    simulated. This helper does not manufacture positive-unlabeled labels.
    """
    directory = DEFAULT_DATA_HOME if data_home is None else data_home
    records = _bike_records(directory, download_if_missing)
    return _digits_gallery(), _bike_gallery(records)
