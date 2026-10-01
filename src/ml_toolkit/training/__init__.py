"""Reusable model training workflow; projects supply data and settings."""

from .evaluation import DiagnosticsOptions
from .results import CrossValidationResult, TrainingResult, TuningResult
from .train import train_model

__all__ = ["CrossValidationResult", "DiagnosticsOptions", "TrainingResult", "TuningResult", "train_model"]
