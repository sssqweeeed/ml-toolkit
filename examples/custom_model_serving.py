"""MLflow Models From Code entrypoint for the target-scaled example."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict, cast

import catboost
import mlflow
import numpy as np
import pandas as pd
from mlflow.pyfunc.model import PythonModel, PythonModelContext
from numpy.typing import NDArray


class Settings(TypedDict):
    target_scale: float
    feature_names: list[str]
    cat_features: list[str]


class TargetScaledServingModel(PythonModel):
    def load_context(self, context: PythonModelContext) -> None:
        self.settings = cast(
            Settings, json.loads(Path(context.artifacts["settings"]).read_text())
        )
        self.model = catboost.CatBoostRegressor()
        self.model.load_model(context.artifacts["model"])

    def predict(
        self,
        context: PythonModelContext | None,
        model_input: pd.DataFrame,
        params: dict[str, object] | None = None,
    ) -> NDArray[np.float64]:
        """Return original-unit predictions after native categorical preparation.

        Parameters
        ----------
        context : PythonModelContext or None
            MLflow serving context; unused during prediction.
        model_input : pandas.DataFrame
            Fitted feature columns; categorical values are converted to strings.
        params : dict or None, default None
            Unused inference parameters.

        Returns
        -------
        numpy.ndarray
            One original-unit prediction per input row.
        """
        frame = model_input.loc[:, self.settings["feature_names"]].copy()
        for feature in self.settings["cat_features"]:
            frame[feature] = frame[feature].astype("string").fillna("unknown").astype(str)
        return np.asarray(self.model.predict(frame), dtype=np.float64) * self.settings["target_scale"]


mlflow.models.set_model(TargetScaledServingModel())
