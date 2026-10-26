from __future__ import annotations

import logging
from datetime import date
from typing import TYPE_CHECKING, Any, Self, cast

import catboost
import mlflow
import numpy as np
import pandas as pd
import polars as pl
from mlflow.models.model import ModelInfo
from mlflow.pyfunc.model import PythonModel

from ..inflation import InflationAdjuster, inflation_from_config
from ..tracking.state import InflationState
from ..types import FloatArray, ModelParams
from ._catboost import read_state
from .catboost_regression import CatBoostLogLinkRegressionModel, CatBoostRegressionModel

if TYPE_CHECKING:
    from ..config import TrackingConfig
    from ..training.contracts import Pool

logger = logging.getLogger(__name__)


class InflationAdjustedModel(PythonModel):
    """Train at one price level and predict at each observation date.

    Parameters
    ----------
    model : CatBoostRegressionModel or None, default None
        Wrapped regression adapter fitted on inflation-normalized targets.
    inflation : InflationAdjuster or None, default None
        Strategy converting monetary targets and predictions between dates.
    reference_date : datetime.date or None, default None
        Common price date used to normalize targets before fitting.
    prediction_date_column : str, default 'prediction_date'
        Serving input column containing requested price dates. Must not
        overwrite a fitted model feature when logging.

    Notes
    -----
    Training data must supply with_target. Pool predictions use data.time;
    frame predictions use prediction_date_column. All three construction values
    may be omitted for artifact loading; partial configuration is rejected.
    """

    def __init__(
        self,
        model: CatBoostRegressionModel | None = None,
        inflation: InflationAdjuster | None = None,
        reference_date: date | None = None,
        prediction_date_column: str = "prediction_date",
    ) -> None:
        super().__init__()
        if any(
            value is None for value in (model, inflation, reference_date)
        ) and not all(value is None for value in (model, inflation, reference_date)):
            raise ValueError("Supply model, inflation and reference_date together")
        self.model = model
        self.inflation = inflation
        self.reference_date = reference_date
        self.prediction_date_column = prediction_date_column

    @property
    def native_model(self) -> catboost.CatBoostRegressor:
        """Return the wrapped fitted CatBoost regressor."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted")
        return self.model.native_model

    @property
    def best_iteration(self) -> int | None:
        """Return the wrapped zero-based best iteration index, or None if unavailable."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted")
        return self.model.best_iteration

    def feature_importances(self, data: Pool) -> dict[str, float]:
        """Delegate importance ranking to the fitted regressor.

        Parameters
        ----------
        data : Pool
            Current feature subset.

        Returns
        -------
        dict[str, float]
            Underlying regression feature importances.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted")
        return self.model.feature_importances(data)

    def get_refit_params(self) -> ModelParams:
        """Delegate final-fit settings to the fitted underlying regressor.

        Returns
        -------
        ModelParams
            Copied underlying constructor parameters. A fresh factory recreates
            and fits the inflation strategy and data-dependent baseline.

        Raises
        ------
        RuntimeError
            The underlying model is absent or has not been fitted.
        """
        if self.model is None:
            raise RuntimeError("Configure model before obtaining refit parameters")
        return self.model.get_refit_params()

    def fit(self, train: Pool, validation: Pool | None = None) -> Self:
        """Fit inflation on training rows and regress targets at the reference date.

        Parameters
        ----------
        train : Pool
            Project-owned training data supplying the conversion methods required
            by this model.
        validation : Pool or None, default None
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
        logger.debug(
            "Fitting inflation-adjusted model at reference date %s", self.reference_date
        )
        if self.model is None or self.inflation is None or self.reference_date is None:
            raise RuntimeError(
                "Configure model, inflation and reference_date before fitting"
            )
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

    def predict_pool(self, data: Pool) -> FloatArray:
        """Predict monetary values at each row's observation date.

        Parameters
        ----------
        data : Pool
            Project data containing fitted feature columns and Date/Datetime values
            in time.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Reference-date regression predictions adjusted to data.time.
        """
        frame = data.to_model_input()
        if self.prediction_date_column in frame.columns:
            raise ValueError(
                "prediction_date_column must not overwrite a model feature"
            )
        frame[self.prediction_date_column] = pd.Series(
            data.time.to_list(),
            index=frame.index,
            dtype="datetime64[ns]",
        )
        return self.predict_frame(frame)

    def predict_frame(
        self,
        model_input: pl.DataFrame | pd.DataFrame,
    ) -> FloatArray:
        """Predict monetary values at dates supplied in a serving frame.

        Parameters
        ----------
        model_input : polars.DataFrame or pandas.DataFrame
            Fitted feature columns plus prediction_date_column with parseable,
            non-null dates.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Date-adjusted predictions; an empty array for an empty frame with the
            required date column.

        Raises
        ------
        TypeError
            model_input is not a Polars or pandas DataFrame.
        ValueError
            The date column is missing, null, unparseable, or outside the inflation
            strategy range.
        """
        if self.prediction_date_column not in model_input.columns:
            raise ValueError(
                f"Missing model input column: {self.prediction_date_column}"
            )
        if len(model_input) == 0:
            return np.asarray([], dtype=float)

        logger.debug(
            "Predicting with inflation-adjusted model: %s rows", len(model_input)
        )
        dates = model_input[self.prediction_date_column]
        if isinstance(dates, pl.Series):
            dates = dates.to_pandas()
        prediction_dates = pd.to_datetime(dates, errors="raise")
        if prediction_dates.isna().any():
            raise ValueError(
                f"{self.prediction_date_column} must not contain null values"
            )

        assert (
            self.model is not None
            and self.inflation is not None
            and self.reference_date is not None
        )
        prediction = self.model.predict_frame(model_input)
        if prediction.shape != (len(model_input),):
            raise ValueError("Inflation adjustment requires one prediction per row")
        return self.inflation.adjust(
            values=prediction,
            from_dates=self._reference_dates(len(model_input)),
            to_dates=pl.from_pandas(
                prediction_dates.rename(self.prediction_date_column)
            ),
        )

    def predict(
        self, context: Any, model_input: pd.DataFrame, params: ModelParams | None = None
    ) -> FloatArray:
        """Predict through MLflow using the date-aware frame API.

        Parameters
        ----------
        context : Any
            MLflow context; unused.
        model_input : pandas.DataFrame
            Fitted features and prediction dates.
        params : dict or None, default None
            Unused inference parameters.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Predictions adjusted to the requested dates.
        """
        return self.predict_frame(model_input)

    def log(
        self,
        sample: Pool,
        *,
        config: TrackingConfig | None = None,
    ) -> tuple[ModelInfo, ModelInfo]:
        """Log native and inflation-adjusted serving models to MLflow.

        Parameters
        ----------
        sample : Pool
            Rows used to build a serving example including observation dates.
        config : TrackingConfig or None, default None
            Artifact name, positive example-row limit, registry name, and serving
            environment. None uses TrackingConfig defaults.

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
        from ..config import TrackingConfig
        from ..tracking._model_state import log_catboost

        config = TrackingConfig() if config is None else config
        name = config.model_name
        registered_model_name = config.registered_model_name
        conda_env = config.conda_env
        logger.debug("Packaging inflation-adjusted model %s", name)
        assert (
            self.model is not None
            and self.inflation is not None
            and self.reference_date is not None
        )
        inflation_config = self.inflation.to_config()
        inflation_from_config(inflation_config)
        if self.prediction_date_column in self.model.feature_names:
            raise ValueError(
                "prediction_date_column must not overwrite a model feature"
            )
        from dataclasses import replace

        native_info = self.model.log(
            sample,
            config=replace(
                config,
                model_name=f"{name}_native",
                registered_model_name=None,
            ),
        )

        sample_frame = sample.to_model_input(
            n_rows=config.input_example_rows, categorical_as_strings=True
        ).loc[:, self.model.feature_names]
        sample_frame[self.prediction_date_column] = pd.Series(
            sample.time.head(len(sample_frame)).to_list(),
            dtype="datetime64[ns]",
            index=sample_frame.index,
        )
        sample_prediction = self.predict(None, sample_frame)

        wrapper_info = log_catboost(
            self,
            sample_frame,
            sample_prediction,
            name=name,
            registered_model_name=registered_model_name,
            conda_env=conda_env,
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

    def _at_reference_date(self, data: Pool) -> Pool:
        assert (
            self.model is not None
            and self.inflation is not None
            and self.reference_date is not None
        )
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

    @property
    def task(self) -> str:
        """Return the regression task identifier."""
        return "regression"

    def export_state(self) -> InflationState:
        assert (
            self.model is not None
            and self.inflation is not None
            and self.reference_date is not None
        )
        return {
            "version": 1,
            "model_type": type(self).__name__,
            "baseline_mode": "nested_model",
            "regressor": self.model.export_state(),
            "inflation": self.inflation.to_config(),
            "reference_date": self.reference_date.isoformat(),
            "prediction_date_column": self.prediction_date_column,
        }

    def load_context(self, context: Any) -> None:
        state = cast(
            InflationState, read_state(context, type(self).__name__, "nested_model")
        )
        types = {
            cls.__name__: cls
            for cls in (CatBoostRegressionModel, CatBoostLogLinkRegressionModel)
        }
        regressor = state["regressor"]
        if regressor.get("version") != 1 or regressor["model_type"] not in types:
            raise ValueError("Unsupported nested regression state")
        self.model = types[regressor["model_type"]]()
        if regressor.get("baseline_mode") != self.model.baseline_mode:
            raise ValueError("Invalid nested regression baseline mode")
        self.model.restore_state(regressor, context.artifacts["model"])
        self.inflation = inflation_from_config(state["inflation"])
        self.reference_date = date.fromisoformat(state["reference_date"])
        self.prediction_date_column = state["prediction_date_column"]
