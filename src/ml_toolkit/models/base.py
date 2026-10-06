from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, Self

import numpy as np

if TYPE_CHECKING:
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
    def native_model(self) -> Any:
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

    def predict_pool(self, data: TrainingData) -> np.ndarray:
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

    def log(self, sample: TrainingData, *, config: TrackingConfig | None = None) -> Any:
        """Log a fitted model to the configured MLflow tracking store.

        Parameters
        ----------
        sample : TrainingData
            Data supplying a representative serving input example.
        config : TrackingConfig or None, default None
            Model artifact name, sample size, registry, and environment settings.

        Returns
        -------
        Any
            Backend-specific model logging result.
        """
        ...
