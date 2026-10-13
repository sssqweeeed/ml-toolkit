from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, cast

import catboost
import mlflow
import numpy as np
import pandas as pd
import polars as pl
from mlflow.models.model import ModelInfo
from mlflow.pyfunc.model import PythonModel

from ..tracking.state import RegressionState
from ..types import FloatArray, ModelParams
from ._catboost import (
    feature_importances,
    native_fit,
    normalize_loss,
    prepare_frame,
    read_state,
    refit_parameters,
)

if TYPE_CHECKING:
    from ..config import TrackingConfig
    from ..training.contracts import TrainingData

logger = logging.getLogger(__name__)


class CatBoostRegressionModel(PythonModel):
    """Train and serve a CatBoost regressor with native initialization.

    Parameters
    ----------
    params : dict or None, default None
        Copied native parameters. ``loss`` is an alias for ``loss_function``;
        native and custom objectives are accepted without a toolkit whitelist.
    prediction_type : str or None, default None
        Native output type; None preserves CatBoost's loss-dependent default.

    Attributes
    ----------
    baseline : float or None
        Manual initial raw score, or None with native initialization.
    """

    baseline_mode = "native_initialization"

    def __init__(
        self,
        params: ModelParams | None = None,
        prediction_type: str | None = None,
    ) -> None:
        self.params = {} if params is None else params.copy()
        self.prediction_type = prediction_type
        self._model = None
        self._baseline = None

    @property
    def task(self) -> str:
        """Return the regression task identifier."""
        return "regression"

    @property
    def baseline(self) -> float | None:
        """Return the fitted manual baseline, or None for native initialization."""
        self.native_model
        return self._baseline

    @property
    def native_model(self) -> catboost.CatBoostRegressor:
        """Return the fitted native CatBoost model; raise RuntimeError before fitting."""
        if self._model is None:
            raise RuntimeError("Model has not been fitted")
        return self._model

    @property
    def best_iteration(self) -> int | None:
        """Return the zero-based best iteration index, or None when unavailable."""
        value = self.native_model.get_best_iteration()
        if value is None or value < 0:
            return None
        return int(value)

    def feature_importances(self, data: TrainingData) -> dict[str, float]:
        """Rank the current fitted feature columns.

        Parameters
        ----------
        data : TrainingData
            Rows with the fitted feature names in order.

        Returns
        -------
        dict[str, float]
            Native prediction-change importance per feature.
        """
        return feature_importances(self.native_model, data.X.columns)

    def get_refit_params(self) -> ModelParams:
        """Return copied constructor parameters for final fitting.

        Returns
        -------
        ModelParams
            The best iteration selected on validation plus one, when available;
            otherwise the original iteration budget. Early stopping is removed
            and use_best_model is disabled. Learned baselines are recalculated.

        Raises
        ------
        RuntimeError
            The adapter has not been fitted.
        """
        self.native_model
        params = refit_parameters(self.params, self.best_iteration)
        return params

    @property
    def feature_names(self) -> list[str]:
        """Return fitted feature names in native model order."""
        names = self.native_model.feature_names_
        if not names:
            raise RuntimeError("Fitted model does not contain feature names")
        return list(names)

    @property
    def cat_features(self) -> list[str]:
        """Return categorical feature names in native model order."""
        return [
            self.feature_names[index]
            for index in self.native_model.get_cat_feature_indices()
        ]

    @property
    def numerical_features(self) -> list[str]:
        """Return fitted feature names not marked categorical."""
        categorical = set(self.cat_features)
        return [name for name in self.feature_names if name not in categorical]

    def fit(self, train: TrainingData, validation: TrainingData | None = None) -> Self:
        """Fit the native regressor using this class's initialization policy.

        Parameters
        ----------
        train : TrainingData
            Training features, targets and weights.
        validation : TrainingData or None, default None
            Optional holdout for early stopping. It uses the training baseline.

        Returns
        -------
        Self
            Fitted model, replacing previous state. Manual baselines are folded
            into native bias once; native initialization is left unchanged.
        """
        logger.debug(
            "Fitting CatBoost regressor: %s training rows, validation=%s",
            len(train),
            validation is not None,
        )
        params = normalize_loss(self.params)
        self._validate_params(params)
        baseline = self._calculate_baseline(train)
        if baseline is not None:
            params["boost_from_average"] = False
        model = native_fit(
            catboost.CatBoostRegressor,
            params,
            train.to_catboost(baseline=baseline),
            None if validation is None else validation.to_catboost(baseline=baseline),
        )
        if baseline is not None:
            scale, bias = model.get_scale_and_bias()
            model.set_scale_and_bias(scale, float(bias) + baseline)
        self._model, self._baseline = model, baseline
        logger.debug("Fitted CatBoost regressor with baseline %s", baseline)
        return self

    def _validate_params(self, params: ModelParams) -> None:
        pass

    def _calculate_baseline(self, train: TrainingData) -> float | None:
        return None

    def predict_frame(self, data: pl.DataFrame | pd.DataFrame) -> FloatArray:
        """Predict with native output semantics and preserve output dimensions.

        Parameters
        ----------
        data : polars.DataFrame or pandas.DataFrame
            Fitted feature columns; extras are ignored and columns reordered.

        Returns
        -------
        numpy.ndarray
            Native predictions, potentially multidimensional. The training
            workflow accepts only a single numeric prediction per row.
        """
        logger.debug("Predicting with CatBoost regressor: %s rows", len(data))
        frame = prepare_frame(data, self.feature_names, self.cat_features)
        options = (
            {}
            if self.prediction_type is None
            else {"prediction_type": self.prediction_type}
        )
        return np.asarray(self.native_model.predict(frame, **options), dtype=float)

    def predict_pool(self, data: TrainingData) -> FloatArray:
        """Predict native regression outputs for project data.

        Parameters
        ----------
        data : TrainingData
            Project data whose X contains all fitted model features.

        Returns
        -------
        numpy.ndarray
            Native predictions with their original output dimensions.

        Raises
        ------
        RuntimeError
            The model has not been fitted.
        """
        return self.predict_frame(data.to_model_input())

    def predict(
        self, context: Any, model_input: pd.DataFrame, params: ModelParams | None = None
    ) -> FloatArray:
        """Predict through the MLflow PythonModel interface.

        Parameters
        ----------
        context : Any
            MLflow context; unused.
        model_input : pandas.DataFrame
            Serving frame containing fitted feature names.
        params : dict or None, default None
            MLflow inference parameters; unused.

        Returns
        -------
        numpy.ndarray
            Native predictions with their original output dimensions.

        See Also
        --------
        predict_frame : Predict directly from a Polars or pandas feature frame.
        """
        return self.predict_frame(model_input)

    def log(
        self,
        sample: TrainingData,
        *,
        config: TrackingConfig | None = None,
    ) -> ModelInfo:
        """Package and log the fitted model with a serving input example.

        Parameters
        ----------
        sample : TrainingData
            Serving example rows, capped by config.input_example_rows.
        config : TrackingConfig or None, default None
            Artifact name, positive example-row limit, registry name, and serving
            environment. None uses TrackingConfig defaults.

        Returns
        -------
        mlflow.models.ModelInfo
            Logged model information, including its model URI.

        Notes
        -----
        Writes native model and configuration artifacts, infers a serving
        signature, and logs baseline and input-schema metadata to MLflow. Configure
        tracking and an experiment before calling.
        """
        from ..config import TrackingConfig

        config = TrackingConfig() if config is None else config
        name = config.model_name
        registered_model_name = config.registered_model_name
        conda_env = config.conda_env
        sample_frame = sample.to_model_input(n_rows=config.input_example_rows).loc[
            :, self.feature_names
        ]
        model_info = self.log_artifacts(
            sample_frame,
            self.predict_frame(sample_frame),
            name=name,
            registered_model_name=registered_model_name,
            conda_env=conda_env,
        )
        mlflow.log_param("baseline", self.baseline)
        mlflow.log_dict(
            {
                "numerical_features": self.numerical_features,
                "cat_features": self.cat_features,
            },
            "model_metadata/catboost_input_schema.json",
        )
        return model_info

    def log_artifacts(
        self,
        sample_frame: pd.DataFrame,
        sample_prediction: FloatArray,
        *,
        name: str,
        registered_model_name: str | None = None,
        conda_env: Path | None = None,
        metadata: dict[str, object] | None = None,
    ) -> ModelInfo:
        """Package the full model class, native state and serving example.

        Parameters
        ----------
        sample_frame : pandas.DataFrame
            Nonempty serving input example.
        sample_prediction : array-like
            Predictions aligned with the example.
        name : str
            MLflow artifact name.
        registered_model_name : str or None, default None
            Optional registry name.
        conda_env : path-like or None, default None
            Explicit environment; otherwise installed dependencies are pinned.
        metadata : dict or None, default None
            Additional audit metadata.

        Returns
        -------
        mlflow.models.ModelInfo
            Logged model information.
        """
        from ..tracking._model_state import log_catboost

        return log_catboost(
            self,
            sample_frame,
            sample_prediction,
            name=name,
            registered_model_name=registered_model_name,
            conda_env=conda_env,
            metadata=metadata,
        )

    def export_state(self) -> RegressionState:
        return {
            "version": 1,
            "model_type": type(self).__name__,
            "prediction_type": self.prediction_type,
            "baseline": self.baseline,
            "baseline_mode": self.baseline_mode,
            "training_params": self.params.copy(),
        }

    def restore_state(self, state: RegressionState, model_path: str) -> None:
        self._model = catboost.CatBoostRegressor()
        self._model.load_model(model_path)
        self.params = state["training_params"]
        self.prediction_type = state["prediction_type"]
        self._baseline = state["baseline"]

    def load_context(self, context: Any) -> None:
        state = cast(
            RegressionState,
            read_state(context, type(self).__name__, self.baseline_mode),
        )
        self.restore_state(state, context.artifacts["model"])
        logger.debug("Loaded CatBoost regressor: %s", type(self).__name__)


class CatBoostLogLinkRegressionModel(CatBoostRegressionModel):
    """Initialize Poisson/Tweedie regression with log(weighted target mean).

    Parameters
    ----------
    params : dict or None, default None
        Native parameters; loss must be Poisson or Tweedie. Defaults to Poisson.
    prediction_type : str or None, default None
        Native output type, inheriting the loss-dependent default when omitted.

    Notes
    -----
    The weighted target mean must be finite and positive. The same baseline
    is supplied to training and validation and folded into native bias once.
    """

    baseline_mode = "native_bias"

    def _validate_params(self, params: ModelParams) -> None:
        loss = params.setdefault("loss_function", "Poisson")
        if not isinstance(loss, str) or loss.split(":", 1)[0] not in (
            "Poisson",
            "Tweedie",
        ):
            raise ValueError("Log-link regression requires Poisson or Tweedie loss")

    def _calculate_baseline(self, train: TrainingData) -> float:
        mean = float(
            np.average(
                train.y.to_numpy(),
                weights=train.weight.to_numpy(),
            )
        )
        if not np.isfinite(mean) or mean <= 0:
            raise ValueError("Weighted target mean must be finite and positive")
        return float(np.log(mean))
