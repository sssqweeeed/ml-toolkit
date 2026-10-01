"""Standalone serving code for CatBoost, with optional inflation adjustment."""

import json
from datetime import date

import catboost
import mlflow
import numpy as np
import pandas as pd
import polars as pl


class CatBoostWrapper(mlflow.pyfunc.PythonModel):
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

    def predict(self, context, model_input, params=None):
        features = model_input.loc[:, self.native_model.feature_names_]
        if model_input.empty:
            return np.asarray([], dtype=float)
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
