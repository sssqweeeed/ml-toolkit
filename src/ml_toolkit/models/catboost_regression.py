from __future__ import annotations

from typing import Any, Self

import catboost
import mlflow
import numpy as np
import pandas as pd
import polars as pl

from .._console import Logger
from ..data import Pool, validate_training_categories
from ..types import FloatArray, ModelParams
from ._catboost import CatBoostAdapter, native_fit, normalize_loss
from .state import RegressionState

logger = Logger(__name__)


class CatBoostRegressionModel(
    CatBoostAdapter[catboost.CatBoostRegressor, RegressionState]
):
    """Train and serve a CatBoost regressor with native initialization.

    Parameters
    ----------
    params : dict or None, default None
        Copied native parameters. ``loss`` is an alias for ``loss_function``;
        native and custom objectives are accepted without a toolkit whitelist.
        ``Gamma`` selects the toolkit's compiled positive-target objective.
    prediction_type : str or None, default None
        Native output type; None preserves CatBoost's loss-dependent default,
        or selects Exponent for the toolkit's Gamma objective.

    Attributes
    ----------
    baseline : float or None
        Manual initial raw score, or None with native initialization.
    """

    _native_type = catboost.CatBoostRegressor
    baseline_mode = "native_initialization"

    def __init__(
        self,
        params: ModelParams | None = None,
        prediction_type: str | None = None,
    ) -> None:
        super().__init__(params)
        self.prediction_type = prediction_type
        self._effective_prediction_type = prediction_type

    @property
    def task(self) -> str:
        """Return the regression task identifier."""
        return "regression"

    @property
    def baseline(self) -> float | None:
        """Return the fitted manual baseline, or None for native initialization."""
        self._require_fitted()
        return self._baseline

    def fit(self, train: Pool, validation: Pool | None = None) -> Self:
        """Fit the native regressor using this class's initialization policy.

        Parameters
        ----------
        train : Pool
            Raw categorical values must not stringify to the reserved "unknown".
            Training features, targets and weights.
        validation : Pool or None, default None
            Subject to the same reserved-category check as training.
            Optional holdout for early stopping. It uses the training baseline.

        Returns
        -------
        Self
            Fitted model, replacing previous state. Manual baselines are folded
            into native bias once; native initialization is left unchanged.
        """
        self._retained_iterations = None
        logger.debug(
            "Fitting CatBoost regressor: %s training rows, validation=%s",
            len(train),
            validation is not None,
        )
        validate_training_categories(train)
        if validation is not None:
            validate_training_categories(validation)
        params = normalize_loss(self.params)
        self._validate_params(params)
        loss = params.get("loss_function")
        is_gamma = isinstance(loss, str) and loss == "Gamma"
        if is_gamma:
            # Training-only compiled code stays out of serving environments.
            from .losses.gamma import configure_gamma

            configure_gamma(
                params, train, validation, log_link=self.baseline_mode == "native_bias"
            )
        baseline = self._calculate_baseline(train)
        if baseline is not None:
            params["boost_from_average"] = False
        model = native_fit(
            self._native_type,
            params,
            train.to_catboost(baseline=baseline),
            None if validation is None else validation.to_catboost(baseline=baseline),
        )
        if baseline is not None:
            scale, bias = model.get_scale_and_bias()
            model.set_scale_and_bias(scale, float(bias) + baseline)
        if is_gamma:
            for split in (train, validation):
                if split is not None:
                    values = np.asarray(
                        model.predict(
                            split.to_model_input(), prediction_type="Exponent"
                        )
                    )
                    if not np.isfinite(values).all() or (values <= 0).any():
                        raise ValueError(
                            "Gamma training produced non-finite or non-positive means"
                        )
        self._model, self._baseline = model, baseline
        self._effective_prediction_type = (
            "Exponent"
            if is_gamma and self.prediction_type is None
            else self.prediction_type
        )
        logger.debug("Fitted CatBoost regressor with baseline %s", baseline)
        return self

    def _validate_params(self, params: ModelParams) -> None:
        pass

    def _calculate_baseline(self, train: Pool) -> float | None:
        return None

    def predict_frame(self, data: pl.DataFrame | pd.DataFrame) -> FloatArray:
        """Predict with native output semantics and preserve output dimensions.

        Parameters
        ----------
        data : polars.DataFrame or pandas.DataFrame
            Fitted feature columns; extras are ignored and columns reordered.
            Categorical columns are formatted exactly as in training. Pandas
            integer categoricals with missing values must use the nullable
            ``Int64`` dtype; a ``float64`` column formats as ``"1.0"``.
            Object-dtype categorical columns are converted with pandas string
            formatting.

        Returns
        -------
        numpy.ndarray
            Native predictions, potentially multidimensional. The training
            workflow accepts only a single numeric prediction per row.
        """
        logger.debug("Predicting with CatBoost regressor: %s rows", len(data))
        frame = self._input_frame(data)
        options = (
            {}
            if self._effective_prediction_type is None
            else {"prediction_type": self._effective_prediction_type}
        )
        return np.asarray(self._require_fitted().predict(frame, **options), dtype=float)

    def predict_pool(self, data: Pool) -> FloatArray:
        """Predict native regression outputs for project data.

        Parameters
        ----------
        data : Pool
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
        return self.predict_frame(data.X)

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

    def export_state(self) -> RegressionState:
        return {
            "version": 1,
            "model_type": type(self).__name__,
            "prediction_type": self.prediction_type,
            "effective_prediction_type": self._effective_prediction_type,
            "baseline": self.baseline,
            "baseline_mode": self.baseline_mode,
            "training_params": self.params.copy(),
            "retained_iterations": self._retained_iterations,
        }

    def _log_fitted_parameters(self) -> None:
        mlflow.log_param("baseline", self.baseline)

    def _restore_fitted_fields(self, state: RegressionState) -> None:
        self.prediction_type = state["prediction_type"]
        self._effective_prediction_type = state.get(
            "effective_prediction_type", self.prediction_type
        )


class CatBoostLogLinkRegressionModel(CatBoostRegressionModel):
    """Initialize Poisson/Tweedie/Gamma regression with log(weighted target mean).

    Parameters
    ----------
    params : dict or None, default None
        Native parameters; loss is Poisson, Tweedie or Gamma. Defaults to Poisson.
        Gamma requires finite, strictly positive targets and supports CPU or
        one CUDA GPU. Its metric is Gamma deviance unless explicitly overridden.
    prediction_type : str or None, default None
        Native output type, inheriting the loss-dependent default when omitted.
        Gamma defaults to Exponent; RawFormulaVal explicitly returns log means.

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
            "Gamma",
        ):
            raise ValueError(
                "Log-link regression requires Poisson or Tweedie or Gamma loss"
            )

    def _calculate_baseline(self, train: Pool) -> float:
        mean = float(
            np.average(
                train.y.to_numpy(),
                weights=train.weight.to_numpy(),
            )
        )
        if not np.isfinite(mean) or mean <= 0:
            raise ValueError("Weighted target mean must be finite and positive")
        return float(np.log(mean))
