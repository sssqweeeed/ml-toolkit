"""Metadata-only validation results and the final fitted model."""

from dataclasses import dataclass

import numpy as np

from ..feature_selection import FeatureSelectionResult
from ..models.base import Model
from ..types import ModelParams


@dataclass(frozen=True)
class SplitResult:
    """Store one complete train/validation execution without retaining its model.

    Attributes
    ----------
    index : int
        Positive unique index supplied by the splitter.
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
    """Collect split results in iterator order and their validation averages.

    Attributes
    ----------
    splits : tuple[SplitResult, ...]
        Evaluated splits in splitter iterator order, without fitted models.
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
            Last validation result in the splitter iterator's order.

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
        Fresh model fitted on the training pool returned by splitter.get_final().
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
        Final model's test scores; None when the final split has no held-out sample.
    """

    model: Model
    validation: ValidationResult
    params: ModelParams
    features: tuple[str, ...]
    feature_selection: FeatureSelectionResult | None = None
    tuning: TuningResult | None = None
    test_metrics: dict[str, float] | None = None
