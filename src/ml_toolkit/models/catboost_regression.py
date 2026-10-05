from __future__ import annotations

import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Self

import catboost
import mlflow
import numpy as np
import pandas as pd
import polars as pl

from .artifacts import log_model_from_code

if TYPE_CHECKING:
    from ..training.contracts import TrainingData

logger = logging.getLogger(__name__)


class CatBoostModel(mlflow.pyfunc.PythonModel):  # type: ignore
    """CatBoost regression model with a constant log-scale baseline."""

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params.copy()
        self._model: catboost.CatBoostRegressor | None = None
        self._baseline: float | None = None

    @property
    def native_model(self) -> catboost.CatBoostRegressor:
        if self._model is None:
            raise RuntimeError("Model has not been fitted")
        return self._model

    @property
    def baseline(self) -> float:
        if self._baseline is None:
            raise RuntimeError("Model has not been fitted")
        return self._baseline

    @property
    def best_iteration(self) -> int | None:
        value = self.native_model.get_best_iteration()
        if value is None or value < 0:
            return None
        return int(value)

    @property
    def feature_names(self) -> list[str]:
        names = self.native_model.feature_names_
        if not names:
            raise RuntimeError("Fitted model does not contain feature names")
        return list(names)

    def fit(self, train: TrainingData, validation: TrainingData | None = None) -> Self:
        logger.debug(
            "Fitting CatBoost regressor: %s training rows, validation=%s",
            len(train),
            validation is not None,
        )
        baseline = self._calculate_baseline(train)
        params = self.params.copy()
        if validation is None:
            params.pop("early_stopping_rounds", None)
            params["use_best_model"] = False

        model = catboost.CatBoostRegressor(**params)
        train_data = train.to_catboost(baseline=baseline)
        if validation is None:
            model.fit(train_data)
        else:
            model.fit(
                train_data,
                eval_set=validation.to_catboost(baseline=baseline),
            )

        scale, bias = model.get_scale_and_bias()
        model.set_scale_and_bias(float(scale), float(bias) + baseline)

        self._model = model
        self._baseline = baseline
        logger.debug("Fitted CatBoost regressor with baseline %s", baseline)
        return self

    def predict_pool(self, data: TrainingData) -> np.ndarray:
        return self.predict_frame(data.to_model_input())

    def select_features(
        self,
        train: TrainingData,
        validation: TrainingData,
        num_features_to_select: int,
    ) -> list[str]:
        feature_names = train.X.columns
        if not 0 < num_features_to_select <= len(feature_names):
            raise ValueError(
                "num_features_to_select must be between 1 and the feature count"
            )
        if num_features_to_select == len(feature_names):
            logger.debug("CatBoost feature selection retained all %s features", len(feature_names))
            return feature_names

        logger.debug(
            "Selecting %s of %s CatBoost features",
            num_features_to_select,
            len(feature_names),
        )
        baseline = self._calculate_baseline(train)
        model = catboost.CatBoostRegressor(**self.params)
        summary = model.select_features(
            train.to_catboost(baseline=baseline),
            eval_set=validation.to_catboost(baseline=baseline),
            features_for_select=feature_names,
            num_features_to_select=num_features_to_select,
            steps=min(50, len(feature_names) - num_features_to_select),
            algorithm=catboost.EFeaturesSelectionAlgorithm.RecursiveByShapValues,
            shap_calc_type=catboost.EShapCalcType.Regular,
            train_final_model=False,
            plot=False,
            logging_level="Verbose",
        )
        names = list(summary["selected_features_names"])
        logger.debug("CatBoost selected features: %s", names)
        return names

    def predict(self, context, model_input, params=None) -> np.ndarray:
        return self.predict_frame(model_input)

    @property
    def cat_features(self) -> list[str]:
        return [
            self.feature_names[index]
            for index in self.native_model.get_cat_feature_indices()
        ]

    @property
    def numerical_features(self) -> list[str]:
        categorical = set(self.cat_features)
        return [name for name in self.feature_names if name not in categorical]

    def log(
        self,
        sample: TrainingData,
        *,
        name="catboost_model",
        registered_model_name=None,
        conda_env=None,
    ) -> Any:
        sample_frame = sample.to_model_input(n_rows=5).loc[:, self.feature_names]
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
        sample_frame,
        sample_prediction,
        *,
        name,
        registered_model_name=None,
        conda_env=None,
        metadata=None,
        code_paths=None,
    ) -> Any:
        metadata = {
            "numerical_features": self.numerical_features,
            "cat_features": self.cat_features,
            "baseline": self.baseline,
            **(metadata or {}),
        }
        with TemporaryDirectory(prefix="catboost-mlflow-") as directory:
            logger.debug("Packaging CatBoost regression model %s", name)
            model_path = Path(directory) / "model.cbm"
            config_path = Path(directory) / "config.json"
            self.native_model.save_model(str(model_path))
            config_path.write_text(json.dumps(metadata), encoding="utf-8")
            return log_model_from_code(
                source=Path(__file__).with_name("catboost_wrapper.py").resolve(),
                sample_frame=sample_frame,
                sample_prediction=sample_prediction,
                artifacts={"model": str(model_path), "config": str(config_path)},
                metadata=metadata,
                params=self.native_model.get_all_params(),
                name=name,
                registered_model_name=registered_model_name,
                conda_env=conda_env,
                code_paths=code_paths,
                packages=("catboost", "polars", "pyarrow"),
            )

    def predict_frame(self, data: pl.DataFrame | pd.DataFrame) -> np.ndarray:
        if isinstance(data, pl.DataFrame):
            feature_frame: pl.DataFrame | pd.DataFrame = data.select(self.feature_names)
        elif isinstance(data, pd.DataFrame):
            missing = sorted(set(self.feature_names) - set(data.columns))
            if missing:
                raise ValueError(f"Missing model input columns: {missing}")
            feature_frame = data.loc[:, self.feature_names]
        else:
            raise TypeError("Prediction data must be a Polars or Pandas DataFrame")

        logger.debug("Predicting with CatBoost regressor: %s rows", len(feature_frame))
        return np.asarray(
            self.native_model.predict(
                feature_frame,
                prediction_type="Exponent",
            ),
            dtype=float,
        ).reshape(-1)

    @staticmethod
    def _calculate_baseline(data: TrainingData) -> float:
        mean = float(
            np.average(
                data.y.to_numpy(),
                weights=data.weight.to_numpy(),
            )
        )
        if not np.isfinite(mean) or mean <= 0:
            raise ValueError("Weighted target mean must be finite and positive")
        return float(np.log(mean))
