"""Data-only reports, independent of model predictions."""

from ._common import (
    data_comparison_pair as data_comparison_pair,
)
from .base import DataAnalyzer
from .distributions import NumericalFeatureDistributions

__all__ = ["DataAnalyzer", "NumericalFeatureDistributions"]
