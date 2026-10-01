from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import catboost
import mlflow
import numpy as np
import pandas as pd
import polars as pl

from ..inflation import InflationAdjuster, inflation_from_config
from .catboost_regression import CatBoostModel

if TYPE_CHECKING:
    from ..training.contracts import TrainingData


class InflationAdjustedModel(mlflow.pyfunc.PythonModel):  # type: ignore
    """CatBoost model trained at one price level and predicted at row dates."""

    def __init__(
        self,
        model: CatBoostModel,
        inflation: InflationAdjuster,
        reference_date: date,
        prediction_date_column: str = "prediction_date",
    ) -> None:
        super().__init__()
        self.model = model
        self.inflation = inflation
        self.reference_date = reference_date
        self.prediction_date_column = prediction_date_column

    @property
    def native_model(self) -> catboost.CatBoostRegressor:
        return self.model.native_model

    @property
    def best_iteration(self) -> int | None:
        return self.model.best_iteration

    def fit(self, train: TrainingData, validation: TrainingData | None = None) -> Self:
        self.inflation.fit(
            dates=train.time,
            y=train.y.to_numpy(),
            sample_weight=train.weight.to_numpy(),
        )
        normalized_train = self._at_reference_date(train)
        normalized_validation = (
            None if validation is None else self._at_reference_date(validation)
        )
        self.model.fit(normalized_train, normalized_validation)
        return self

    def predict_pool(self, data: TrainingData) -> np.ndarray:
        prediction = self.model.predict_pool(data)
        return self.inflation.adjust(
            values=prediction,
            from_dates=self._reference_dates(len(data)),
            to_dates=data.time,
        )

    def select_features(
        self,
        train: TrainingData,
        validation: TrainingData,
        num_features_to_select: int,
    ) -> list[str]:
        self.inflation.fit(
            dates=train.time,
            y=train.y.to_numpy(),
            sample_weight=train.weight.to_numpy(),
        )
        return self.model.select_features(
            train=self._at_reference_date(train),
            validation=self._at_reference_date(validation),
            num_features_to_select=num_features_to_select,
        )

    def predict(  # type: ignore
        self,
        context: Any,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> np.ndarray:
        del context, params
        if not isinstance(model_input, pd.DataFrame):
            raise TypeError("model_input must be a pandas.DataFrame")
        if self.prediction_date_column not in model_input.columns:
            raise ValueError(
                f"Missing model input column: {self.prediction_date_column}"
            )
        if model_input.empty:
            return np.asarray([], dtype=float)

        prediction_dates = pd.to_datetime(
            model_input[self.prediction_date_column],
            errors="raise",
        )
        if prediction_dates.isna().any():
            raise ValueError(
                f"{self.prediction_date_column} must not contain null values"
            )

        prediction = self.model.predict_frame(model_input)
        return self.inflation.adjust(
            values=prediction,
            from_dates=self._reference_dates(len(model_input)),
            to_dates=pl.Series(
                self.prediction_date_column,
                prediction_dates.dt.date.to_list(),
                dtype=pl.Date,
            ),
        )

    def log(
        self,
        sample: TrainingData,
        *,
        name="inflation_adjusted_model",
        registered_model_name=None,
        conda_env=None,
    ) -> tuple[Any, Any]:
        inflation_config = self.inflation.to_config()
        inflation_from_config(inflation_config)
        if self.prediction_date_column in self.model.feature_names:
            raise ValueError(
                "prediction_date_column must not overwrite a model feature"
            )
        native_info = self.model.log(sample, conda_env=conda_env)

        sample_frame = sample.to_model_input(n_rows=5).loc[:, self.model.feature_names]
        sample_frame[self.prediction_date_column] = pd.Series(
            sample.time.head(len(sample_frame)).to_list(),
            dtype="datetime64[ns]",
        )
        sample_prediction = self.predict(None, sample_frame)

        wrapper_info = self.model.log_artifacts(
            sample_frame,
            sample_prediction,
            name=name,
            registered_model_name=registered_model_name,
            conda_env=conda_env,
            code_paths=[str(Path(__file__).resolve().parents[1] / "inflation.py")],
            metadata={
                "reference_date": self.reference_date.isoformat(),
                "prediction_date_column": self.prediction_date_column,
                "inflation_strategy": type(self.inflation).__name__,
                "inflation": inflation_config,
            },
        )

        mlflow.log_param("reference_date", self.reference_date.isoformat())
        mlflow.log_param("prediction_date_column", self.prediction_date_column)
        mlflow.log_param("inflation_strategy", type(self.inflation).__name__)
        mlflow.log_dict(
            self.inflation.to_config(),
            "model_metadata/inflation.json",
        )
        return native_info, wrapper_info

    def _at_reference_date(self, data: TrainingData) -> TrainingData:
        return data.with_target(
            self.inflation.adjust(
                values=data.y.to_numpy(),
                from_dates=data.time,
                to_dates=self._reference_dates(len(data)),
            )
        )

    def _reference_dates(self, length: int) -> pl.Series:
        return pl.Series(
            "reference_date",
            [self.reference_date] * length,
            dtype=pl.Date,
        )
