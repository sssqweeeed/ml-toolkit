"""Log and load native LightGBM regressors with MLflow."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Self

import lightgbm
import numpy as np
import pandas as pd
from mlflow.models.model import ModelInfo
from mlflow.pyfunc.model import PythonModel

from .._console import Logger
from ..types import FloatArray, ModelParams

if TYPE_CHECKING:
    from ..config import TrackingConfig
    from ..training.contracts import Pool

logger = Logger("ml_toolkit.models.lightgbm")


class LightGBMServingModel(PythonModel):
    """Log and serve a LightGBM regressor without pickling the model.

    Parameters
    ----------
    model : lightgbm.Booster or fitted estimator or None, default None
        Native Booster or fitted scikit-learn-style LightGBM estimator. None is
        used when MLflow loads artifacts.
    numerical_features : Sequence[str] or None, default None
        Numerical feature names; None uses an empty list.
    cat_features : Sequence[str] or None, default None
        Categorical feature names; None uses an empty list.

    Notes
    -----
    Requires the lightgbm extra and a working native installation. Numerical
    and categorical names must partition native model features when logging.
    """

    def __init__(
        self,
        model: lightgbm.Booster | lightgbm.LGBMModel | None = None,
        numerical_features: Sequence[str] | None = None,
        cat_features: Sequence[str] | None = None,
    ) -> None:
        self.model = model
        self.numerical_features = (
            list(numerical_features) if numerical_features is not None else []
        )
        self.cat_features = list(cat_features) if cat_features is not None else []

    @property
    def n_iterations(self) -> int:
        """Return the iteration count used by the serving prediction default.

        A positive best_iteration limits ordinary LightGBM predictions even
        when the Booster retains additional iterations.
        """
        native = self.native_model
        available = native.current_iteration()
        return (
            min(native.best_iteration, available)
            if native.best_iteration > 0
            else available
        )

    def shrink(self, n_iterations: int) -> Self:
        """Return an independent Booster prefix without fitting new trees."""
        from ._shrink import validate_size

        validate_size(n_iterations, self.n_iterations)
        native = self.native_model
        reduced = lightgbm.Booster(
            model_str=native.model_to_string(
                start_iteration=0, num_iteration=n_iterations
            )
        )
        # Exclude the estimator too: its training state is not part of serving.
        result = deepcopy(self, {id(self.model): reduced})
        result.model = reduced
        return result

    def predict_pool(self, data: Pool) -> FloatArray:
        """Predict pool rows using the same feature preparation as serving."""
        return self.predict(None, data.to_model_input())

    @property
    def task(self) -> str:
        """Return the regression task identifier."""
        return "regression"

    @property
    def native_model(self) -> lightgbm.Booster:
        """Return the native Booster; raise RuntimeError if no model is loaded."""
        if self.model is None:
            raise RuntimeError("The LightGBM model has not been loaded.")
        return (
            self.model
            if isinstance(self.model, lightgbm.Booster)
            else self.model.booster_
        )

    @property
    def feature_names(self) -> list[str]:
        """Return native LightGBM feature names in prediction order."""
        return self.native_model.feature_name()

    def load_context(self, context: Any) -> None:
        self.model = lightgbm.Booster(model_file=context.artifacts["booster"])
        with open(context.artifacts["features"], encoding="utf-8") as f:
            features = json.load(f)
        self.numerical_features = features["numerical_features"]
        self.cat_features = features["cat_features"]
        logger.debug("Loaded LightGBM serving model")

    def predict(
        self, context: Any, model_input: pd.DataFrame, params: ModelParams | None = None
    ) -> FloatArray:
        """Predict with native feature order and pandas categorical conversion.

        Parameters
        ----------
        context : Any
            MLflow context; unused.
        model_input : pandas.DataFrame
            Frame containing all native feature columns. Configured categorical
            columns are cast to pandas category dtype.
        params : dict or None, default None
            MLflow inference parameters; unused.

        Returns
        -------
        numpy.ndarray
            Native Booster predictions.

        Raises
        ------
        RuntimeError
            No model is loaded.
        """
        if self.model is None:
            raise RuntimeError("The LightGBM model has not been loaded.")
        X = pd.DataFrame(
            {name: model_input[name] for name in self.feature_names},
            index=model_input.index,
            copy=False,
        )
        for feature in self.cat_features:
            X[feature] = X[feature].astype("category")
        logger.debug("Predicting with LightGBM serving model: %s rows", len(X))
        return np.asarray(self.native_model.predict(X))

    def log(
        self,
        sample: Pool,
        *,
        config: TrackingConfig | None = None,
        extra_artifacts: Mapping[str, str] | None = None,
    ) -> ModelInfo:
        """Log native LightGBM artifacts and a serving signature to MLflow.

        Parameters
        ----------
        sample : Pool
            Project data supplying at least one serving row via to_model_input; up
            to config.input_example_rows rows are used.
        config : TrackingConfig or None, default None
            Artifact name, serving example row count, registry, and environment.
            None uses TrackingConfig defaults, including no registration.

        extra_artifacts : Mapping[str, str] or None, default None
            Additional files or directories bundled with the model. Names must
            not conflict with the adapter's own artifacts.

        Returns
        -------
        mlflow.models.ModelInfo
            Logged model information.

        Raises
        ------
        ValueError
            Feature metadata does not partition native model features or the
            serving sample is empty.

        Notes
        -----
        Writes native Booster and feature artifacts with a separate registration
        entry point and package sources. Configure MLflow before calling.
        """
        from ..config import TrackingConfig
        from ..tracking.model_artifacts import log_model_from_code

        config = TrackingConfig() if config is None else config
        name = config.model_name
        registered_model_name = config.registered_model_name
        conda_env = config.conda_env

        features = self.numerical_features + self.cat_features
        if len(features) != len(set(features)) or set(features) != set(
            self.feature_names
        ):
            raise ValueError(
                "numerical_features and cat_features must partition the model features."
            )

        wrapper_path = (
            Path(__file__).resolve().parents[1]
            / "tracking"
            / "entrypoints"
            / "LightGBMServingModel.py"
        )
        sample_frame = sample.to_model_input(
            n_rows=config.input_example_rows, categorical_as_strings=True
        ).loc[:, self.feature_names]
        if sample_frame.empty:
            raise ValueError("The logging sample must contain at least one row.")
        sample_prediction = self.predict(None, sample_frame)
        booster = self.native_model
        metadata: dict[str, object] = {
            "numerical_features": self.numerical_features,
            "cat_features": self.cat_features,
        }

        with TemporaryDirectory(prefix="lightgbm-mlflow-") as tmp:
            logger.debug("Packaging LightGBM model %s", name)
            booster_path = Path(tmp) / "model.lgb"
            features_path = Path(tmp) / "features.json"
            booster.save_model(str(booster_path))
            features_path.write_text(json.dumps(metadata), encoding="utf-8")

            model_info = log_model_from_code(
                source=wrapper_path,
                name=name,
                registered_model_name=registered_model_name,
                sample_frame=sample_frame,
                sample_prediction=sample_prediction,
                params=booster.params,
                conda_env=conda_env,
                artifacts={
                    "booster": str(booster_path),
                    "features": str(features_path),
                },
                metadata=metadata,
                extra_artifacts=extra_artifacts,
                packages=("lightgbm",),
            )

        return model_info
