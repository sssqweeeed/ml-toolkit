"""Metadata-only validation results and the final fitted model."""

from dataclasses import dataclass

import numpy as np

from ..models.base import Model
from ..types import ModelParams


@dataclass(frozen=True)
class FeatureSelectionStep:
    """Store one fitted and evaluated feature subset.

    Attributes
    ----------
    features : tuple[str, ...]
        Evaluated input columns in original order.
    metrics : dict[str, float]
        All supplied validation scores returned by the training metric.
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
    Validation scores describe selection, not independent performance.
    Frozen result objects contain mutable metric and importance dictionaries.
    """

    features: tuple[str, ...]
    selected_step: int
    steps: tuple[FeatureSelectionStep, ...]


@dataclass(frozen=True)
class SplitResult:
    """Store one complete train/validation execution without retaining its model.

    Attributes
    ----------
    index : int
        One-based original split index before skipping.
    train_metrics : dict[str, float]
        Metrics on the training rows.
    val_metrics : dict[str, float]
        Metrics on held-out validation rows.
    features : tuple[str, ...]
        Selected feature names in input order.
    refit_params : ModelParams
        Constructor parameters for fitting a new model without validation.
    best_iteration : int or None
        Zero-based best iteration of the post-tuning fit, not an RFE iteration.
    feature_selection : FeatureSelectionResult or None
        Selection history from this split before tuning.
    tuning : TuningResult or None
        Winning parameters and objective of this split's independent study.
    """

    index: int
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]
    features: tuple[str, ...]
    refit_params: ModelParams
    best_iteration: int | None
    feature_selection: FeatureSelectionResult | None = None
    tuning: "TuningResult | None" = None


@dataclass(frozen=True)
class ValidationResult:
    """Collect chronological split results and their validation averages.

    Attributes
    ----------
    splits : tuple[SplitResult, ...]
        Evaluated splits in chronological order, without fitted models.
    """

    splits: tuple[SplitResult, ...]

    @property
    def mean_metrics(self) -> dict[str, float]:
        """Return the unweighted mean of each validation metric over splits.

        Returns
        -------
        dict[str, float]
            Named means; a single holdout returns its own validation metrics.

        Raises
        ------
        ValueError
            No splits exist or metric names differ between splits.
        """
        if not self.splits:
            raise ValueError("Validation result contains no splits")
        names = set(self.splits[0].val_metrics)
        if any(set(split.val_metrics) != names for split in self.splits):
            raise ValueError("Validation metric names must match across splits")
        return {
            name: float(np.mean([split.val_metrics[name] for split in self.splits]))
            for name in self.splits[0].val_metrics
        }

    @property
    def last_split(self) -> SplitResult:
        """Return the latest evaluated split, used for final fitting.

        Returns
        -------
        SplitResult
            Last chronological validation result.

        Raises
        ------
        ValueError
            No validation splits exist.
        """
        if not self.splits:
            raise ValueError("Validation result contains no splits")
        return self.splits[-1]


@dataclass(frozen=True)
class TuningResult:
    """Store the winning parameters and objective of one Optuna study.

    Attributes
    ----------
    best_params : ModelParams
        Complete effective constructor parameters, including fixed overrides.
    best_score : float
        Finite objective returned by the winning score callback.
    best_trial : int
        Zero-based Optuna trial number.
    """

    best_params: ModelParams
    best_score: float
    best_trial: int


@dataclass(frozen=True)
class TrainingResult:
    """Store the final fitted model and independent split workflows' results.

    Attributes
    ----------
    model : Model
        Fresh model fitted on all rows before test.
    validation : ValidationResult
        Per-split selection, tuning and post-tuning evaluation results.
    params : ModelParams
        Final constructor parameters from the last validation split.
    features : tuple[str, ...]
        Features from the last validation split, in input order.
    feature_selection : FeatureSelectionResult or None
        Last validation split's selection history.
    tuning : TuningResult or None
        Last validation split's optional optimization result.
    test_metrics : dict[str, float] or None
        Final model's test scores; None when test_size is None.
    """

    model: Model
    validation: ValidationResult
    params: ModelParams
    features: tuple[str, ...]
    feature_selection: FeatureSelectionResult | None = None
    tuning: TuningResult | None = None
    test_metrics: dict[str, float] | None = None
