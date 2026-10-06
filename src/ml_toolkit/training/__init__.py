"""Reusable model training workflow; projects supply data and settings."""

from .contracts import (
    DatasetSplit,
    IdentifiedTrainingData,
    PostFitArtifactWriter,
    PostFitEvaluator,
)
from ..config import (
    FeatureSelectionConfig,
    CrossValidationConfig,
    TuningConfig,
    FinalFitConfig,
    PSIOptions,
    FigureConfig,
    NumericalFeatureDistributionConfig,
    ShapConfig,
    CommonPlotsConfig,
    ClassificationPlotsConfig,
    RegressionPlotsConfig,
    PUPlotsConfig,
    ClassificationDiagnosticsConfig,
    RegressionDiagnosticsConfig,
    PUDiagnosticsConfig,
    DiagnosticsConfig,
    TrackingConfig,
    PostFitEvaluationConfig,
    TrainingConfig,
)
from ..config import __all__ as _config_exports
from .results import CrossValidationResult, TrainingResult, TuningResult
from .train import train_model

__all__ = [
    "CrossValidationResult",
    "DatasetSplit",
    "IdentifiedTrainingData",
    "PostFitArtifactWriter",
    "PostFitEvaluator",
    "TrainingResult",
    "TuningResult",
    "train_model",
] + _config_exports
