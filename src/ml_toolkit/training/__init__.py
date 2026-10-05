"""Reusable model training workflow; projects supply data and settings."""

from .contracts import DatasetSplit, IdentifiedTrainingData, PostFitEvaluator
from .evaluation import DiagnosticsOptions
from .results import CrossValidationResult, TrainingResult, TuningResult
from .train import train_model

__all__ = [
    "CrossValidationResult", "DatasetSplit", "DiagnosticsOptions",
    "IdentifiedTrainingData", "PostFitEvaluator", "TrainingResult", "TuningResult",
    "train_model",
]
