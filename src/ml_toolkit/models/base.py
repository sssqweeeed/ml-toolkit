from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self

import catboost
from mlflow.models.model import ModelInfo

from ..types import FloatArray, ModelParams

if TYPE_CHECKING:
    import lightgbm

    from ..config import TrackingConfig
    from ..training.contracts import TrainingData


class Model(Protocol):
    """Describe the model adapter interface consumed by the training workflow.

    Notes
    -----
    Factories receive a parameter dictionary. fit returns the adapter;
    predict_pool returns one finite prediction per row. Models may use any
    native backend.
    """

    @property
    def native_model(self) -> catboost.CatBoost | lightgbm.Booster:
        """Return the underlying fitted backend model for diagnostics."""
        ...

    @property
    def best_iteration(self) -> int | None:
        """Return the zero-based best iteration index, or None when unavailable."""
        ...

    def fit(self, train: TrainingData, validation: TrainingData | None = None) -> Self:
        """Fit the adapter using project-owned training and optional holdout data.

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
            Fitted model adapter.
        """
        ...

    def predict_pool(self, data: TrainingData) -> FloatArray:
        """Predict one numeric value per project data row.

        Parameters
        ----------
        data : TrainingData
            Aligned feature rows accepted by this adapter.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Task-appropriate predictions; classifiers return positive-class
            probabilities.
        """
        ...

    def feature_importances(self, data: TrainingData) -> dict[str, float]:
        """Return feature ranking values for the adapter's backend.

        Parameters
        ----------
        data : TrainingData
            Current feature subset in fitted order.

        Returns
        -------
        dict[str, float]
            Importance for every input feature.
        """
        ...

    def get_refit_params(self) -> ModelParams:
        """Return constructor settings for a fresh fit without validation.

        Returns
        -------
        ModelParams
            Copied parameters containing the fitted iteration budget and any
            selected decision threshold. Early stopping must be disabled.
            Learned model state and data-derived baselines are excluded.
        """
        ...

    def log(
        self, sample: TrainingData, *, config: TrackingConfig | None = None
    ) -> ModelInfo | tuple[ModelInfo, ModelInfo]:
        """Log a fitted model to the configured MLflow tracking store.

        Parameters
        ----------
        sample : TrainingData
            Data supplying a representative serving input example.
        config : TrackingConfig or None, default None
            Model artifact name, sample size, registry, and environment settings.

        Returns
        -------
        ModelInfo or tuple[ModelInfo, ModelInfo]
            Backend-specific model logging result.
        """
        ...
