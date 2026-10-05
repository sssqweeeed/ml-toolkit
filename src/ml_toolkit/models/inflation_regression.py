from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)


class InflationAdjustedModel(mlflow.pyfunc.PythonModel):  # type: ignore
    """Train at one price level and predict at each observation date.

    Parameters
    ----------
    model : CatBoostRegressionModel
        Wrapped regression adapter fitted on inflation-normalized targets.
    inflation : InflationAdjuster
        Strategy converting monetary targets and predictions between dates.
    reference_date : datetime.date
        Common price date used to normalize targets before fitting.
    prediction_date_column : str, default 'prediction_date'
        Serving input column containing requested price dates. Must not
        overwrite a fitted model feature when logging.

    Notes
    -----
    Training data must supply with_target. Pool predictions use data.time;
    pandas serving predictions use prediction_date_column.
    """

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
        """Return the wrapped fitted CatBoost regressor."""
        return self.model.native_model

    @property
    def best_iteration(self) -> int | None:
        """Return the wrapped zero-based best iteration index, or None if unavailable."""
        return self.model.best_iteration

    def fit(self, train: TrainingData, validation: TrainingData | None = None) -> Self:
        """Fit inflation on training rows and regress targets at the reference date.

        Parameters
        ----------
        train : TrainingData
            Project-owned training data supplying the conversion methods required
            by this model.
        validation : TrainingData or None, default None
            Optional holdout data for early stopping and model selection.

        Returns
        -------
        Self
            This adapter with fitted inflation and native regression state.

        Notes
        -----
        Validation targets are normalized using the inflation strategy fitted on
        training data.
        """
        logger.debug("Fitting inflation-adjusted model at reference date %s", self.reference_date)
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
        logger.debug("Fitted inflation-adjusted model")
        return self

    def predict_pool(self, data: TrainingData) -> np.ndarray:
        """Predict monetary values at each row's observation date.

        Parameters
        ----------
        data : TrainingData
            Project data containing fitted feature columns and Date/Datetime values
            in time.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Reference-date regression predictions adjusted to data.time.
        """
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
        """Select regression features after normalizing targets to the reference date.

        Parameters
        ----------
        train : TrainingData
            Data used to train the temporary feature-selection model.
        validation : TrainingData
            Holdout data used to evaluate candidate feature subsets.
        num_features_to_select : int
            Number of features to keep, between one and the available feature
            count.

        Returns
        -------
        list[str]
            Names selected by the wrapped regression adapter.

        Notes
        -----
        Fits the inflation strategy on training rows before converting either
        split.
        """
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
        """Predict monetary values at dates supplied in a pandas serving frame.

        Parameters
        ----------
        context : Any
            MLflow context; unused.
        model_input : pandas.DataFrame
            Fitted feature columns plus prediction_date_column with parseable,
            non-null dates.
        params : dict[str, Any] or None, default None
            MLflow inference parameters; unused.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Date-adjusted predictions; an empty array for an empty frame with the
            required date column.

        Raises
        ------
        TypeError
            model_input is not a pandas DataFrame.
        ValueError
            The date column is missing, null, unparseable, or outside the inflation
            strategy range.
        """
        del context, params
        if not isinstance(model_input, pd.DataFrame):
            raise TypeError("model_input must be a pandas.DataFrame")
        if self.prediction_date_column not in model_input.columns:
            raise ValueError(
                f"Missing model input column: {self.prediction_date_column}"
            )
        if model_input.empty:
            return np.asarray([], dtype=float)

        logger.debug("Predicting with inflation-adjusted model: %s rows", len(model_input))
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
        """Log native and inflation-adjusted serving models to MLflow.

        Parameters
        ----------
        sample : TrainingData
            Rows used to build a serving example including observation dates.
        name : str, default 'inflation_adjusted_model'
            Adjusted model artifact name.
        registered_model_name : str or None, default None
            Optional registry name for the adjusted model.
        conda_env : str, pathlib.Path, or None, default None
            Existing serving environment file; None pins installed versions.

        Returns
        -------
        tuple[mlflow.models.ModelInfo, mlflow.models.ModelInfo]
            Native regression model information followed by adjusted serving model
            information.

        Raises
        ------
        ValueError
            The inflation strategy cannot be reconstructed or the prediction date
            column overlaps a model feature.

        Notes
        -----
        Writes both models, reference-date and strategy parameters, and serialized
        inflation metadata to configured MLflow tracking.
        """
        logger.debug("Packaging inflation-adjusted model %s", name)
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
