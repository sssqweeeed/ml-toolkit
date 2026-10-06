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
    from ..config import FeatureSelectionConfig, TrackingConfig
    from ..training.contracts import TrainingData

logger = logging.getLogger(__name__)


class CatBoostModel(mlflow.pyfunc.PythonModel):  # type: ignore
    """Adapt a CatBoost regressor with a constant log-scale baseline.

    Parameters
    ----------
    params : dict[str, Any]
        CatBoostRegressor constructor parameters, copied on construction.

    Notes
    -----
    Exported as CatBoostRegressionModel from ml_toolkit.models. Training
    initializes raw predictions with log(weighted target mean), which requires
    a finite positive mean. The fitted model bias incorporates this baseline.
    Predictions use CatBoost's Exponent output, yielding positive rates.
    """

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params.copy()
        self._model: catboost.CatBoostRegressor | None = None
        self._baseline: float | None = None

    @property
    def native_model(self) -> catboost.CatBoostRegressor:
        """Return the fitted native CatBoost model; raise RuntimeError before fitting."""
        if self._model is None:
            raise RuntimeError("Model has not been fitted")
        return self._model

    @property
    def baseline(self) -> float:
        """Return the fitted log-mean baseline; raise RuntimeError before fitting."""
        if self._baseline is None:
            raise RuntimeError("Model has not been fitted")
        return self._baseline

    @property
    def best_iteration(self) -> int | None:
        """Return the zero-based best iteration index, or None when unavailable."""
        value = self.native_model.get_best_iteration()
        if value is None or value < 0:
            return None
        return int(value)

    @property
    def feature_names(self) -> list[str]:
        """Return fitted feature names in native model order."""
        names = self.native_model.feature_names_
        if not names:
            raise RuntimeError("Fitted model does not contain feature names")
        return list(names)

    def fit(self, train: TrainingData, validation: TrainingData | None = None) -> Self:
        """Fit a regressor using the training target log-mean baseline.

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
            This fitted adapter, replacing any previous fitted state.

        Raises
        ------
        ValueError
            The exposure-weighted target mean is not finite and positive.

        Notes
        -----
        train.to_catboost and validation.to_catboost must apply the supplied
        baseline and aligned weights. After fitting, the baseline is folded into
        the native model bias. Without validation, early stopping is removed and
        use_best_model is disabled.
        """
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
        """Predict positive rates for project data.

        Parameters
        ----------
        data : TrainingData
            Project data whose X contains all fitted model features.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Positive predictions on the exponentiated raw-model scale.

        Raises
        ------
        RuntimeError
            The model has not been fitted.
        """
        return self.predict_frame(data.to_model_input())

    def select_features(
        self,
        train: TrainingData,
        validation: TrainingData,
        config: FeatureSelectionConfig,
    ) -> list[str]:
        """Select features by recursive SHAP elimination without fitting the adapter.

        Parameters
        ----------
        train : TrainingData
            Data used to train the temporary feature-selection model.
        validation : TrainingData
            Holdout data used to evaluate candidate feature subsets.
        config : FeatureSelectionConfig
            Feature count and backend options. The holdout fraction is used by
            the workflow before calling this method.

        Returns
        -------
        list[str]
            Selected feature names, or all names when the requested count equals
            the available count.

        Raises
        ------
        ValueError
            The requested count is outside the available feature range.

        Notes
        -----
        Uses a temporary native model with train_final_model=False. CatBoost
        selection verbosity follows config.catboost.logging_level.
        """
        num_features_to_select = config.n_features
        feature_names = train.X.columns
        if not 0 < num_features_to_select <= len(feature_names):
            raise ValueError(
                "num_features_to_select must be between 1 and the feature count"
            )
        if num_features_to_select == len(feature_names):
            logger.debug(
                "CatBoost feature selection retained all %s features",
                len(feature_names),
            )
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
            steps=min(
                config.catboost.max_steps, len(feature_names) - num_features_to_select
            ),
            algorithm=catboost.EFeaturesSelectionAlgorithm[config.catboost.algorithm],
            shap_calc_type=catboost.EShapCalcType[config.catboost.shap_calc_type],
            train_final_model=False,
            plot=False,
            logging_level=config.catboost.logging_level,
        )
        names = list(summary["selected_features_names"])
        logger.debug("CatBoost selected features: %s", names)
        return names

    def predict(self, context, model_input: pd.DataFrame, params=None) -> np.ndarray:
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
        numpy.ndarray of shape (n_samples,)
            Positive predictions on the exponentiated raw-model scale.

        See Also
        --------
        predict_frame : Predict directly from a Polars or pandas feature frame.
        """
        return self.predict_frame(model_input)

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

    def log(
        self,
        sample: TrainingData,
        *,
        config: TrackingConfig | None = None,
    ) -> Any:
        """Package and log the fitted model with a serving input example.

        Parameters
        ----------
        sample : TrainingData
            Data used to supply up to config.input_example_rows serving example rows via to_model_input.
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
        sample_frame,
        sample_prediction,
        *,
        name,
        registered_model_name=None,
        conda_env=None,
        metadata=None,
        code_paths=None,
    ) -> Any:
        """Log regression serving code with native artifacts and custom metadata.

        Parameters
        ----------
        sample_frame : pandas.DataFrame
            Nonempty serving example, including any wrapper-specific columns.
        sample_prediction : array-like
            Predictions aligned with sample_frame for signature inference.
        name : str
            MLflow model artifact name.
        registered_model_name : str or None, default None
            Optional registry name.
        conda_env : str, pathlib.Path, or None, default None
            Existing environment file; None pins installed dependency versions.
        metadata : dict or None, default None
            Additional serving configuration merged over feature and baseline
            metadata.
        code_paths : list[str] or None, default None
            Additional source paths bundled by MLflow.

        Returns
        -------
        mlflow.models.ModelInfo
            Logged regression serving model information.

        Notes
        -----
        Packages the fitted model and JSON configuration with CatBoostWrapper
        serving code. Writes artifacts to configured MLflow tracking.
        """
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
        """Predict exponentiated regression outputs from a feature frame.

        Parameters
        ----------
        data : polars.DataFrame or pandas.DataFrame
            Frame containing all fitted feature names. Columns are selected and
            reordered to match the model; extra columns are ignored.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Positive predictions on the exponentiated raw-model scale.

        Raises
        ------
        TypeError
            Input is not a supported DataFrame.
        RuntimeError
            The model has not been fitted.
        """
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
