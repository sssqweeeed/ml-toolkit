"""Shared CatBoost adapter base and the conversions both adapters use."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Generic, Self, TypeVar, cast

import catboost
import mlflow
import numpy as np
import pandas as pd
import polars as pl
from mlflow.models.model import ModelInfo
from mlflow.pyfunc.model import PythonModel

from .._console import Logger
from ..data import Pool, categorical_frame
from ..types import FloatArray, ModelParams
from .artifacts import TrackingConfig, log_catboost
from .base import validate_size
from .state import ClassificationState, RegressionState, read_state

logger = Logger(__name__)

NativeT = TypeVar("NativeT", bound=catboost.CatBoost)
StateT = TypeVar("StateT", RegressionState, ClassificationState)

ITERATION_ALIASES = ("iterations", "n_estimators", "num_trees", "num_boost_round")


def normalize_loss(params: ModelParams) -> ModelParams:
    params = params.copy()
    if "loss" in params:
        loss = params.pop("loss")
        native = params.get("loss_function")
        same_loss = native is loss or (
            isinstance(native, str) and isinstance(loss, str) and native == loss
        )
        if native is not None and not same_loss:
            raise ValueError("Conflicting loss and loss_function values")
        params["loss_function"] = loss
    return params


def prepare_frame(
    data: pl.DataFrame | pd.DataFrame, names: list[str], categorical: list[str]
) -> pd.DataFrame:
    missing = sorted(set(names) - set(data.columns))
    if missing:
        raise ValueError(f"Missing model input columns: {missing}")
    if isinstance(data, pl.DataFrame):
        return categorical_frame(data.select(names), categorical)
    # Construct a new column container without copying numeric buffers, also
    # with pandas versions where selecting columns via .loc makes a deep copy.
    frame = pd.DataFrame(
        {name: data[name] for name in names}, index=data.index, copy=False
    )
    if categorical:
        # Reuse the training conversion so category strings match exactly.
        columns = frame[list(categorical)]
        # Mixed-type object values (also as pandas Categorical dictionaries) cannot
        # convert to Arrow; format them as pandas strings first. They cannot
        # originate from the Polars training path.
        mixed = [
            name
            for name in categorical
            if columns[name].dtype == object
            or (
                isinstance(columns[name].dtype, pd.CategoricalDtype)
                and columns[name].cat.categories.dtype == object
            )
        ]
        columns = columns.astype({name: "string" for name in mixed})
        converted = categorical_frame(pl.from_pandas(columns), categorical)
        converted.index = frame.index
        for name in categorical:
            frame[name] = converted[name]
    return frame


def native_fit(
    model_type: type[NativeT],
    params: ModelParams,
    train_data: catboost.Pool,
    validation_data: catboost.Pool | None,
) -> NativeT:
    params = params.copy()
    if validation_data is None:
        params.pop("early_stopping_rounds", None)
        params["use_best_model"] = False
    model = model_type(**params)
    model.fit(train_data, eval_set=validation_data)
    return model


def refit_parameters(params: ModelParams, best_iteration: int | None) -> ModelParams:
    result = params.copy()
    if best_iteration is not None:
        key = next((name for name in ITERATION_ALIASES if name in result), "iterations")
        result[key] = int(best_iteration) + 1
    for key in ("early_stopping_rounds", "od_type", "od_wait", "od_pval"):
        result.pop(key, None)
    result["use_best_model"] = False
    return result


def feature_importances(
    native: catboost.CatBoost, columns: list[str]
) -> dict[str, float]:
    names = list(native.feature_names_)
    values = np.asarray(
        native.get_feature_importance(type="PredictionValuesChange"), dtype=float
    )
    if not np.isfinite(values).all():
        counts = native.get_tree_leaf_counts()
        leaves = native.get_leaf_values()
        if len(leaves) and leaves.size % int(np.sum(counts)) == 0:
            outputs = leaves.reshape(int(np.sum(counts)), -1)
            start = 0
            constant = True
            for count in counts:
                tree = outputs[start : start + count]
                constant = constant and bool((tree == tree[0]).all())
                start += count
            if constant:
                values = np.zeros(len(names))
    if names != columns or values.shape != (len(names),):
        raise ValueError(
            "Native feature names and dimensions must match the input columns"
        )
    if not np.isfinite(values).all():
        raise ValueError("Native importances must be finite")
    return dict(zip(names, values.tolist(), strict=True))


class CatBoostAdapter(PythonModel, Generic[NativeT, StateT], metaclass=ABCMeta):
    """State and behaviour shared by the CatBoost regression and classification adapters."""

    _native_type: type[NativeT]
    baseline_mode: str

    def __init__(self, params: ModelParams | None = None) -> None:
        self.params = {} if params is None else params.copy()
        self._model: NativeT | None = None
        self._baseline: float | None = None
        self._best_iteration: int | None = None
        self._retained_iterations: int | None = None

    def _require_fitted(self) -> NativeT:
        if self._model is None:
            raise RuntimeError("Model has not been fitted")
        return self._model

    @property
    def native_model(self) -> NativeT:
        """Return the fitted native CatBoost model; raise RuntimeError before fitting."""
        return self._require_fitted()

    @property
    def n_iterations(self) -> int:
        """Return the number of fitted trees used for prediction."""
        return int(self._require_fitted().tree_count_)

    def shrink(self, n_iterations: int) -> Self:
        """Return an independent fitted prefix, preserving prediction scale.

        n_iterations must be an integer from one through the fitted size.
        The copy's refit budget equals its retained size.
        """
        validate_size(n_iterations, self.n_iterations)
        native = self._require_fitted().copy()
        native.shrink(n_iterations, ntree_start=0)
        result = deepcopy(self, {id(self._model): native})
        result._retained_iterations = n_iterations
        return result

    @property
    def best_iteration(self) -> int | None:
        """Return the zero-based best iteration index, or None when unavailable."""
        value = self._best_iteration
        if value is None:
            value = self._require_fitted().get_best_iteration()
        if value is None or value < 0 or value >= self.n_iterations:
            return None
        return int(value)

    def feature_importances(self, data: Pool) -> dict[str, float]:
        """Rank the current fitted feature columns.

        Parameters
        ----------
        data : Pool
            Rows with the fitted feature names in order.

        Returns
        -------
        dict[str, float]
            Native prediction-change importance per feature.
        """
        return feature_importances(self._require_fitted(), data.X.columns)

    def get_refit_params(self) -> ModelParams:
        """Return copied constructor parameters for final fitting.

        Returns
        -------
        ModelParams
            The best iteration selected on validation plus one, when available;
            otherwise the original iteration budget. Early stopping is removed
            and use_best_model is disabled. Learned baselines are recalculated.
            Classifiers include the selected decision threshold when one exists.

        Raises
        ------
        RuntimeError
            The adapter has not been fitted.
        """
        self._require_fitted()
        iteration = (
            self.best_iteration
            if self._retained_iterations is None
            else self._retained_iterations - 1
        )
        return refit_parameters(self.params, iteration)

    @property
    def feature_names(self) -> list[str]:
        """Return fitted feature names in native model order."""
        names = self._require_fitted().feature_names_
        if not names:
            raise RuntimeError("Fitted model does not contain feature names")
        return list(names)

    @property
    def cat_features(self) -> list[str]:
        """Return categorical feature names in native model order."""
        return [
            self.feature_names[index]
            for index in self._require_fitted().get_cat_feature_indices()
        ]

    @property
    def numerical_features(self) -> list[str]:
        """Return fitted feature names not marked categorical."""
        categorical = set(self.cat_features)
        return [name for name in self.feature_names if name not in categorical]

    def _input_frame(self, data: pl.DataFrame | pd.DataFrame) -> pd.DataFrame:
        return prepare_frame(data, self.feature_names, self.cat_features)

    def log(
        self,
        sample: Pool,
        *,
        config: TrackingConfig | None = None,
        extra_artifacts: Mapping[str, str] | None = None,
    ) -> ModelInfo:
        """Package and log the fitted model with a serving input example.

        Parameters
        ----------
        sample : Pool
            Serving example rows, capped by config.input_example_rows.
        config : TrackingConfig or None, default None
            Artifact name, positive example-row limit, registry name, and serving
            environment. None uses TrackingConfig defaults.
        extra_artifacts : Mapping[str, str] or None, default None
            Additional files or directories bundled with the model. Names must
            not conflict with the adapter's own artifacts.

        Returns
        -------
        mlflow.models.ModelInfo
            Logged model information, including its model URI.

        Notes
        -----
        Writes native model and configuration artifacts, infers a serving
        signature, and logs baseline and input-schema metadata to MLflow. Configure
        tracking and an experiment before calling. Classification artifacts also
        preserve threshold and PU settings.
        """
        config = TrackingConfig() if config is None else config
        sample_frame = sample.to_model_input(
            n_rows=config.input_example_rows, categorical_as_strings=True
        ).loc[:, self.feature_names]
        model_info = self._log_artifacts(
            sample_frame,
            self.predict_frame(sample_frame),
            name=config.model_name,
            registered_model_name=config.registered_model_name,
            conda_env=config.conda_env,
            extra_artifacts=extra_artifacts,
        )
        self._log_fitted_parameters()
        mlflow.log_dict(
            {
                "numerical_features": self.numerical_features,
                "cat_features": self.cat_features,
            },
            "model_metadata/catboost_input_schema.json",
        )
        return model_info

    def _log_artifacts(
        self,
        sample_frame: pd.DataFrame,
        sample_prediction: FloatArray,
        *,
        name: str,
        registered_model_name: str | None = None,
        conda_env: Path | None = None,
        metadata: dict[str, object] | None = None,
        extra_artifacts: Mapping[str, str] | None = None,
    ) -> ModelInfo:
        """Package the model and its state as an MLflow model artifact."""
        return log_catboost(
            self,
            sample_frame,
            sample_prediction,
            name=name,
            registered_model_name=registered_model_name,
            conda_env=conda_env,
            extra_artifacts=extra_artifacts,
            metadata=metadata,
        )

    def _restore_state(self, state: StateT, model_path: str) -> None:
        """Load the native model and the persisted adapter state."""
        native = self._native_type()
        native.load_model(model_path)
        self._model = native
        self.params = state["training_params"]
        self._baseline = state["baseline"]
        self._retained_iterations = state.get("retained_iterations")
        self._restore_fitted_fields(state)

    def load_context(self, context: Any) -> None:
        state = cast(
            StateT, read_state(context, type(self).__name__, self.baseline_mode)
        )
        self._restore_state(state, context.artifacts["model"])
        logger.debug("Loaded CatBoost model: %s", type(self).__name__)

    @abstractmethod
    def predict_frame(self, data: pl.DataFrame | pd.DataFrame) -> FloatArray: ...

    @abstractmethod
    def export_state(self) -> StateT: ...

    @abstractmethod
    def _log_fitted_parameters(self) -> None: ...

    @abstractmethod
    def _restore_fitted_fields(self, state: StateT) -> None: ...
