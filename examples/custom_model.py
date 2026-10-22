"""A complete CatBoost-backed Model adapter with target scaling and MLflow serving."""

# --8<-- [start:adapter]
from __future__ import annotations

from ml_toolkit.training import Holdout

import json
from importlib.metadata import version
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Self

import catboost
import mlflow
import numpy as np
import polars as pl
from mlflow.models import infer_signature
from mlflow.models.model import ModelInfo

from ml_toolkit import Pool, train_model
from ml_toolkit.config import TrackingConfig, TrainingConfig
from ml_toolkit.models import CatBoostRegressionModel
from ml_toolkit.types import FloatArray, ModelParams


class TargetScaledRegressor:
    """Fit a CatBoost RMSE regressor in scaled target units.

    Parameters
    ----------
    params : ModelParams
        Native CatBoost RMSE settings, copied by the wrapped adapter.
    target_scale : float, default 1000.0
        Positive finite conversion factor. Training divides targets by this
        factor; prediction multiplies outputs back into original units.

    Notes
    -----
    The factory must reuse the same target_scale for validation and final
    fitting. Native SHAP contributions describe scaled targets. Logged serving
    predictions retain the complete inverse transformation.
    """

    def __init__(self, params: ModelParams, *, target_scale: float = 1000.0) -> None:
        if not isfinite(target_scale) or target_scale <= 0:
            raise ValueError("target_scale must be finite and positive")
        self.target_scale = target_scale
        self._inner = CatBoostRegressionModel(params)

    @property
    def native_model(self) -> catboost.CatBoostRegressor:
        """Return the fitted backend, whose outputs are in scaled target units."""
        return self._inner.native_model

    @property
    def best_iteration(self) -> int | None:
        """Return the zero-based iteration chosen on validation, when available."""
        return self._inner.best_iteration

    def fit(self, train: Pool, validation: Pool | None = None) -> Self:
        """Fit on transformed targets while preserving aligned row metadata.

        Parameters
        ----------
        train : Pool
            Aligned training rows with targets in original units.
        validation : Pool or None, default None
            Optional holdout rows transformed using the same fixed scale.

        Returns
        -------
        Self
            Fitted adapter.
        """
        scaled_train = train.with_target(train.y.cast(pl.Float64) / self.target_scale)
        scaled_validation = (
            None
            if validation is None
            else validation.with_target(
                validation.y.cast(pl.Float64) / self.target_scale
            )
        )
        self._inner.fit(scaled_train, scaled_validation)
        return self

    def predict_pool(self, data: Pool) -> FloatArray:
        """Return one prediction per row in original target units.

        Parameters
        ----------
        data : Pool
            Rows containing the fitted feature columns.

        Returns
        -------
        FloatArray
            Predictions of shape (n_samples,) in original units.
        """
        return self._inner.predict_pool(data) * self.target_scale

    def feature_importances(self, data: Pool) -> dict[str, float]:
        """Return the wrapped backend's importance for each current feature.

        Parameters
        ----------
        data : Pool
            Current feature subset in fitted order.

        Returns
        -------
        dict[str, float]
            Importance for every supplied feature, suitable for importance_ranker.
        """
        return self._inner.feature_importances(data)

    def get_refit_params(self) -> ModelParams:
        """Return fresh constructor settings with the fitted iteration budget.

        Returns
        -------
        ModelParams
            Copied backend parameters with early stopping disabled. The factory
            supplies target_scale separately; learned state is never reused.
        """
        return self._inner.get_refit_params()

    def log(self, sample: Pool, *, config: TrackingConfig | None = None) -> ModelInfo:
        """Log native artifacts and an entrypoint that reverses target scaling.

        Parameters
        ----------
        sample : Pool
            Representative serving rows in fitted feature order.
        config : TrackingConfig or None, default None
            Artifact name, registry, example size, and optional environment file.

        Returns
        -------
        ModelInfo
            Logged complete adapter's serving URI.

        Notes
        -----
        Call inside an active MLflow run. The serving entrypoint uses CatBoost,
        pandas, NumPy, and MLflow; it does not import the training example.
        """
        config = TrackingConfig() if config is None else config
        frame = sample.to_model_input(n_rows=config.input_example_rows)
        frame = frame.loc[:, self._inner.feature_names]
        prediction = self._inner.predict_frame(frame) * self.target_scale
        with TemporaryDirectory(prefix="target-scaled-model-") as directory:
            root = Path(directory)
            model_path = root / "model.cbm"
            self.native_model.save_model(str(model_path))
            state_path = root / "settings.json"
            state_path.write_text(
                json.dumps(
                    {
                        "target_scale": self.target_scale,
                        "feature_names": self._inner.feature_names,
                        "cat_features": self._inner.cat_features,
                    }
                ),
                encoding="utf-8",
            )
            return mlflow.pyfunc.log_model(
                name=config.model_name,
                python_model=str(Path(__file__).with_name("custom_model_serving.py")),
                artifacts={"model": str(model_path), "settings": str(state_path)},
                signature=infer_signature(frame, prediction),
                input_example=frame,
                params={},
                metadata={"target_scale": self.target_scale},
                code_paths=[],
                registered_model_name=config.registered_model_name,
                conda_env=None if config.conda_env is None else str(config.conda_env),
                pip_requirements=(
                    [
                        f"{package}=={version(package)}"
                        for package in ("mlflow", "catboost", "numpy", "pandas")
                    ]
                    if config.conda_env is None
                    else None
                ),
            )


def model_factory(params: ModelParams) -> TargetScaledRegressor:
    """Create an independent adapter using the same project target scale.

    Parameters
    ----------
    params : ModelParams
        Backend settings for a fresh fit.

    Returns
    -------
    TargetScaledRegressor
        Unfitted adapter with a new inner CatBoost model.
    """
    return TargetScaledRegressor(params, target_scale=1000.0)


# --8<-- [end:adapter]


# --8<-- [start:workflow]
from examples.custom_metric import weighted_mae


def demo() -> None:
    """Run deterministic validation, final fitting, and serving parity checks.

    Notes
    -----
    Uses cached UCI Bike Sharing observations and a temporary local MLflow
    store. Prepare the cache with python -m examples.prepare_data, then run
    python -m examples.custom_model. The model fitted without test rows is
    logged and checked on the untouched future test tail; artifacts are removed
    after the check.
    """
    from examples.gallery_data import load_bike_pool

    data = load_bike_pool(download_if_missing=False)
    splitter = Holdout(data, val_size=0.25, test_size=0.2)
    partition = splitter.get_test()
    assert partition.heldout is not None
    params: ModelParams = {
        "iterations": 40,
        "depth": 4,
        "loss_function": "RMSE",
        "random_seed": 42,
        "thread_count": 2,
        "verbose": False,
        "allow_writing_files": False,
    }
    with TemporaryDirectory(prefix="custom-model-demo-") as directory:
        root = Path(directory)
        mlflow.set_tracking_uri(f"sqlite:///{root / 'tracking.db'}")
        experiment = mlflow.create_experiment(
            "custom-model-demo", artifact_location=(root / "artifacts").as_uri()
        )
        mlflow.set_experiment(experiment_id=experiment)
        result = train_model(
            splitter=splitter,
            params=params,
            model_factory=model_factory,
            metric=weighted_mae,
            run_name="scaled-regression",
            config=TrainingConfig(
                tracking=TrackingConfig(log_test_model=False, log_final_model=False)
            ),
        )
        fitted = result.model
        with mlflow.start_run(run_name="serving-parity"):
            logged = fitted.log(partition.train)
        info = logged[1] if isinstance(logged, tuple) else logged
        served = mlflow.pyfunc.load_model(info.model_uri)
        expected = fitted.predict_pool(partition.heldout)
        np.testing.assert_allclose(
            served.predict(partition.heldout.to_model_input()), expected
        )
        native_predictions = fitted.native_model.predict(
            partition.heldout.to_model_input()
        )
        np.testing.assert_allclose(expected, native_predictions * 1000.0)
        print(
            "native predictions (thousands):", native_predictions[:5].round(4).tolist()
        )
        print("observed rentals:", partition.heldout.y.head(5).to_list())
        print("predicted rentals:", expected[:5].round(1).tolist())
        print(f"validation MAE: {result.validation.mean_metrics['mae']:.3f}")
        assert result.test_metrics is not None
        print(f"test MAE: {result.test_metrics['mae']:.3f}")
        print("serving predictions match: True")


# --8<-- [end:workflow]


if __name__ == "__main__":
    demo()
