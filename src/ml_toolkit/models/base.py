from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

import catboost
from mlflow.models.model import ModelInfo

from ..types import FloatArray, ModelParams

if TYPE_CHECKING:
    import lightgbm

    from ..config import TrackingConfig
    from ..training.contracts import Pool


@runtime_checkable
class ShrinkableModel(Protocol):
    """Optional fitted prediction adapter supporting independent prefix copies.

    Sizes count boosting iterations, starting at one, rather than zero-based
    best-iteration indices. Implementing this protocol does not require fit/log.
    """

    @property
    def n_iterations(self) -> int:
        """Return the number of iterations used by ordinary predictions."""
        ...

    def shrink(self, n_iterations: int) -> Self:
        """Return an independent adapter retaining the first n_iterations.

        Reject non-integers and sizes outside [1, n_iterations]. The original
        adapter, its fitted state and predictions must remain unchanged.
        """
        ...

    def predict_pool(self, data: Pool) -> FloatArray:
        """Return task-scale predictions, including fitted transformations."""
        ...


class Model(Protocol):
    """Describe the model adapter interface consumed by the training workflow.

    Notes
    -----
    Factories receive a parameter dictionary. fit returns the adapter;
    predict_pool returns one finite prediction per row. The native_model
    property exposes a fitted CatBoost model or LightGBM Booster.
    """

    @property
    def native_model(self) -> catboost.CatBoost | lightgbm.Booster:
        """Return the underlying fitted backend model for diagnostics."""
        ...

    @property
    def best_iteration(self) -> int | None:
        """Return the zero-based best iteration index, or None when unavailable."""
        ...

    def fit(self, train: Pool, validation: Pool | None = None) -> Self:
        """Fit the adapter using project-owned training and optional holdout data.

        Parameters
        ----------
        train : Pool
            Project-owned training data supplying the conversion methods required
            by this model.
        validation : Pool or None, default None
            Optional holdout data for early stopping and model selection.

        Returns
        -------
        Self
            Fitted model adapter.
        """
        ...

    def predict_pool(self, data: Pool) -> FloatArray:
        """Predict one numeric value per project data row.

        Parameters
        ----------
        data : Pool
            Aligned feature rows accepted by this adapter.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Task-appropriate predictions; classifiers return positive-class
            probabilities.
        """
        ...

    def feature_importances(self, data: Pool) -> dict[str, float]:
        """Return feature ranking values for the adapter's backend.

        Parameters
        ----------
        data : Pool
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
        self,
        sample: Pool,
        *,
        config: TrackingConfig | None = None,
        extra_artifacts: Mapping[str, str] | None = None,
    ) -> ModelInfo | tuple[ModelInfo, ModelInfo]:
        """Log a fitted model to the configured MLflow tracking store.

        Parameters
        ----------
        sample : Pool
            Data supplying a representative serving input example.
        config : TrackingConfig or None, default None
            Model artifact name, sample size, registry, and environment settings.

        extra_artifacts : Mapping[str, str] or None, default None
            Additional files or directories bundled with the model. Names must
            not conflict with the adapter's own artifacts.

        Returns
        -------
        ModelInfo or tuple[ModelInfo, ModelInfo]
            Backend-specific model logging result.
        """
        ...
