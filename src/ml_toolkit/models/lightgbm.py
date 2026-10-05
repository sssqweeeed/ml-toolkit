"""Log and load the legacy LightGBM regressor without pickling it."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import lightgbm
import mlflow
import numpy as np

logger = logging.getLogger("ml_toolkit.models.lightgbm")


class LightGBMWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model=None, numerical_features=None, cat_features=None):
        self.model = model
        self.numerical_features = (
            list(numerical_features) if numerical_features is not None else []
        )
        self.cat_features = list(cat_features) if cat_features is not None else []

    @property
    def native_model(self):
        if self.model is None:
            raise RuntimeError("The LightGBM model has not been loaded.")
        return (
            self.model
            if isinstance(self.model, lightgbm.Booster)
            else self.model.booster_
        )

    @property
    def feature_names(self):
        return self.native_model.feature_name()

    def load_context(self, context):
        self.model = lightgbm.Booster(model_file=context.artifacts["booster"])
        with open(context.artifacts["features"], encoding="utf-8") as f:
            features = json.load(f)
        self.numerical_features = features["numerical_features"]
        self.cat_features = features["cat_features"]
        logger.debug("Loaded LightGBM serving model")

    def predict(self, context, model_input, params=None):
        if self.model is None:
            raise RuntimeError("The LightGBM model has not been loaded.")
        X = model_input.loc[:, self.feature_names].copy()
        for feature in self.cat_features:
            X[feature] = X[feature].astype("category")
        logger.debug("Predicting with LightGBM serving model: %s rows", len(X))
        return np.asarray(self.native_model.predict(X))

    def log(
        self,
        sample,
        *,
        name="severity_osago",
        registered_model_name="severity_osago",
        conda_env=None,
    ):
        from .artifacts import log_model_from_code

        features = self.numerical_features + self.cat_features
        if len(features) != len(set(features)) or set(features) != set(
            self.feature_names
        ):
            raise ValueError(
                "numerical_features and cat_features must partition the model features."
            )

        wrapper_path = Path(__file__).resolve()
        sample_frame = sample.to_model_input(n_rows=5).loc[:, self.feature_names]
        if sample_frame.empty:
            raise ValueError("The logging sample must contain at least one row.")
        sample_prediction = self.predict(None, sample_frame)
        booster = self.native_model
        metadata = {
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
                packages=("lightgbm",),
            )

        return model_info


mlflow.models.set_model(LightGBMWrapper())
