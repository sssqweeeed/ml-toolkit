"""Structured outputs from cross-validation, tuning, and final fitting."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..models.base import Model


@dataclass(frozen=True)
class CrossValidationResult:
    fold_metrics: tuple[dict[str, float], ...]
    best_iterations: tuple[int, ...]

    @property
    def mean_metrics(self) -> dict[str, float]:
        if not self.fold_metrics:
            raise ValueError("Cross-validation result contains no folds")
        return {
            key: float(np.mean([metrics[key] for metrics in self.fold_metrics]))
            for key in self.fold_metrics[0]
        }

    @property
    def final_iterations(self) -> int | None:
        return None if not self.best_iterations else int(np.median(self.best_iterations)) + 1


@dataclass(frozen=True)
class TuningResult:
    best_params: dict[str, Any]
    cross_validation: CrossValidationResult


@dataclass(frozen=True)
class TrainingResult:
    model: Model
    cross_validation: CrossValidationResult
    params: dict[str, Any]
    features: tuple[str, ...]
