"""Standalone serving code for CatBoost, with optional inflation adjustment."""

import json
import logging
from datetime import date

import catboost
import mlflow
import numpy as np
import pandas as pd
import polars as pl

logger = logging.getLogger("ml_toolkit.models.catboost_wrapper")


class CatBoostWrapper(mlflow.pyfunc.PythonModel):
    """Serve a logged CatBoost regressor with optional inflation adjustment.

    Notes
    -----
    MLflow loads native model and JSON configuration artifacts through
    load_context. Feature columns follow the native model order; optional
    inflation configuration supplies the reference and prediction dates.
    """

    def load_context(self, context):
        self.native_model = catboost.CatBoostRegressor()
        self.native_model.load_model(context.artifacts["model"])
        with open(context.artifacts["config"], encoding="utf-8") as handle:
            self.config = json.load(handle)
        self.numerical_features = self.config["numerical_features"]
        self.cat_features = self.config["cat_features"]
        self.inflation = None
        if "inflation" in self.config:
            from inflation import inflation_from_config

            self.inflation = inflation_from_config(self.config["inflation"])

        logger.debug("Loaded CatBoost serving model: inflation=%s", self.inflation is not None)

    def predict(self, context, model_input, params=None):
        """Predict exponentiated outputs with optional row-date inflation adjustment.

        Parameters
        ----------
        context : Any
            MLflow context; unused during prediction.
        model_input : pandas.DataFrame
            Native feature columns plus the configured prediction date column when
            inflation is enabled.
        params : dict or None, default None
            MLflow inference parameters; unused.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Positive regression outputs, optionally expressed at requested row
            dates; an empty array for an empty input with required features.
        """
        features = model_input.loc[:, self.native_model.feature_names_]
        if model_input.empty:
            return np.asarray([], dtype=float)
        logger.debug("Predicting with CatBoost serving model: %s rows", len(features))
        prediction = np.asarray(
            self.native_model.predict(features, prediction_type="Exponent"),
            dtype=float,
        ).reshape(-1)
        if self.inflation is None:
            return prediction

        date_column = self.config["prediction_date_column"]
        dates = pd.to_datetime(model_input[date_column], errors="raise")
        if dates.isna().any():
            raise ValueError(f"{date_column} must not contain null values")
        return self.inflation.adjust(
            values=prediction,
            from_dates=pl.Series(
                "reference_date",
                [date.fromisoformat(self.config["reference_date"])] * len(features),
                dtype=pl.Date,
            ),
            to_dates=pl.Series(date_column, dates.dt.date.to_list(), dtype=pl.Date),
        )


mlflow.models.set_model(CatBoostWrapper())
