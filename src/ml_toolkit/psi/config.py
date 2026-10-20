"""Configuration for standalone PSI analysis."""

from dataclasses import dataclass

from ..types import TimePeriod
from .._time import PERIOD_INTERVALS
from ..feature_levels import FeatureLevelConfig


@dataclass(frozen=True)
class PSIConfig:
    """Configure reference-based population stability analysis.

    Parameters
    ----------
    features : tuple[FeatureLevelConfig, ...]
        Nonempty collection of uniquely named feature definitions, normalized
        to a tuple.
    period : TimePeriod, default "month"
        Calendar period for temporal distributions, preserving the input timezone.
    n_bins : int, default 10
        Positive requested number of numerical quantile bins. Ties and constant
        features may produce fewer bins.
    epsilon : float, default 1e-6
        Share floor strictly between zero and one, applied before logarithms.
    """

    features: tuple[FeatureLevelConfig, ...]
    period: TimePeriod = "month"
    n_bins: int = 10
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.period not in PERIOD_INTERVALS:
            raise ValueError("Unsupported time period")
        object.__setattr__(self, "features", tuple(self.features))
        if not self.features:
            raise ValueError("features must not be empty")
        names = [feature.name for feature in self.features]
        if len(names) != len(set(names)):
            raise ValueError("Feature names must be unique")
        if self.n_bins < 1:
            raise ValueError("n_bins must be a positive integer")
        if not 0 < self.epsilon < 1:
            raise ValueError("epsilon must be between 0 and 1")
