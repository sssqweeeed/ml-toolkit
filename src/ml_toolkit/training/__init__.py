"""Reusable model training workflow; projects supply data and settings."""

from .contracts import (
    DatasetSplit,
    FeatureRanker,
    IdentifiedTrainingData,
    PostFitArtifactWriter,
    PostFitEvaluator,
)
from ..config import (
    RFEFeatureSelectionConfig,
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
from .results import (
    CrossValidationResult, FeatureSelectionResult, FeatureSelectionStep,
    TrainingResult, TuningResult,
)
from .feature_selection import select_features
from .rankers import ShapRanker, importance_ranker
from .train import train_model

__all__ = [
    "CrossValidationResult",
    "DatasetSplit",
    "FeatureRanker",
    "FeatureSelectionResult",
    "FeatureSelectionStep",
    "ShapRanker",
    "importance_ranker",
    "select_features",
    "IdentifiedTrainingData",
    "PostFitArtifactWriter",
    "PostFitEvaluator",
    "TrainingResult",
    "TuningResult",
    "train_model",
] + _config_exports
