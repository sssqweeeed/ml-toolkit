"""Structured outputs from cross-validation, tuning, and final fitting."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..models.base import Model


@dataclass(frozen=True)
class FeatureSelectionStep:
    """Store one fitted and evaluated feature subset.

    Attributes
    ----------
    features : tuple[str, ...]
        Evaluated input columns in original order.
    metrics : dict[str, float]
        All internal validation scores returned by the training metric.
    importances : dict[str, float]
        Ranking values by feature name; empty when no elimination follows.
    eliminated_features : tuple[str, ...]
        Columns removed before the next fit, in original column order.
    fit_seconds : float
        Elapsed adapter construction and training time.
    evaluation_seconds : float
        Elapsed prediction and metric calculation time.
    ranking_seconds : float
        Elapsed ranking and importance validation time, or zero on the last step.
    best_iteration : int or None
        Adapter's zero-based best iteration; not used for final fitting.
    """

    features: tuple[str, ...]
    metrics: dict[str, float]
    importances: dict[str, float]
    eliminated_features: tuple[str, ...]
    fit_seconds: float
    evaluation_seconds: float
    ranking_seconds: float
    best_iteration: int | None


@dataclass(frozen=True)
class FeatureSelectionResult:
    """Store batch RFE's selected subset and complete search history.

    Attributes
    ----------
    features : tuple[str, ...]
        Selected columns in original input order; may include the full set.
    selected_step : int
        Zero-based index into steps for the selected subset.
    steps : tuple[FeatureSelectionStep, ...]
        Chronological history of every fitted subset; contains no fitted models.

    Notes
    -----
    Internal validation scores describe selection, not independent performance.
    Frozen result objects contain mutable metric and importance dictionaries.
    """

    features: tuple[str, ...]
    selected_step: int
    steps: tuple[FeatureSelectionStep, ...]


@dataclass(frozen=True)
class CrossValidationResult:
    """Store validation metrics and best iteration indices from evaluated folds.

    Attributes
    ----------
    fold_metrics : tuple[dict[str, float], ...]
        One validation metric dictionary per evaluated fold, in fold order.
    best_iterations : tuple[int, ...]
        Zero-based best iteration indices from folds that report them; may be
        shorter than fold_metrics.

    Examples
    --------
    >>> from ml_toolkit.training import CrossValidationResult
    >>> result = CrossValidationResult(({"mae": 2.0}, {"mae": 4.0}), (9, 19))
    >>> result.mean_metrics
    {'mae': 3.0}
    >>> result.final_iterations
    15
    """

    fold_metrics: tuple[dict[str, float], ...]
    best_iterations: tuple[int, ...]

    @property
    def mean_metrics(self) -> dict[str, float]:
        """Return the unweighted mean of each validation metric across folds.

        Returns
        -------
        dict[str, float]
            Means for keys in the first fold; all folds must contain those keys.

        Raises
        ------
        ValueError
            The result contains no evaluated folds.
        """
        if not self.fold_metrics:
            raise ValueError("Cross-validation result contains no folds")
        return {
            key: float(np.mean([metrics[key] for metrics in self.fold_metrics]))
            for key in self.fold_metrics[0]
        }

    @property
    def final_iterations(self) -> int | None:
        """Return the median best iteration index plus one, or None if unavailable."""
        return (
            None
            if not self.best_iterations
            else int(np.median(self.best_iterations)) + 1
        )


@dataclass(frozen=True)
class TuningResult:
    """Store the winning Optuna parameters and validation result.

    Attributes
    ----------
    best_params : dict[str, Any]
        Parameter overrides returned by param_space for the winning trial,
        including fixed callback values and excluding unchanged base settings.
    cross_validation : CrossValidationResult
        Fold results recorded for that trial.
    """

    best_params: dict[str, Any]
    cross_validation: CrossValidationResult


@dataclass(frozen=True)
class TrainingResult:
    """Store the fitted model and the settings used for final training.

    Attributes
    ----------
    model : Model
        Model fitted on all rows with the retained features.
    cross_validation : CrossValidationResult
        Validation metrics from the selected parameter configuration.
    params : dict[str, Any]
        Final model parameters, including the selected iteration count when
        available.
    features : tuple[str, ...]
        Retained feature names in data column order.
    feature_selection : FeatureSelectionResult or None
        Internal RFE history, or None when selection was disabled.
    """

    model: Model
    cross_validation: CrossValidationResult
    params: dict[str, Any]
    features: tuple[str, ...]
    feature_selection: FeatureSelectionResult | None = None
