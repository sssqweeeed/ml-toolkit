"""Configuration and fitted level definitions for PSI analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

FeatureType = Literal["categorical", "numerical"]

NULL_LEVEL = "__NULL__"
ERROR_LEVEL = "__ERROR__"
UNSEEN_LEVEL = "__UNSEEN__"


@dataclass(frozen=True)
class PSIFeature:
    name: str
    ftype: FeatureType
    error_level: Any | None = None
    null_level: Any | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Feature name must not be empty")
        if self.ftype not in ("categorical", "numerical"):
            raise ValueError(f"Unknown feature type: {self.ftype!r}")
        if isinstance(self.error_level, (float, np.floating)) and np.isnan(self.error_level):
            raise ValueError("error_level must not be NaN")
        if self.error_level is not None and self.null_level is not None:
            if self.error_level == self.null_level:
                raise ValueError("error_level and null_level must be different")
        if self.ftype == "categorical" and (
            self.lower_bound is not None or self.upper_bound is not None
        ):
            raise ValueError("Bounds are supported only for numerical features")
        if self.lower_bound is not None and (
            np.isnan(self.lower_bound) or self.lower_bound == np.inf
        ):
            raise ValueError("lower_bound must be a number or -inf")
        if self.upper_bound is not None and (
            np.isnan(self.upper_bound) or self.upper_bound == -np.inf
        ):
            raise ValueError("upper_bound must be a number or +inf")
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound > self.upper_bound
        ):
            raise ValueError("lower_bound must not exceed upper_bound")


@dataclass(frozen=True)
class PSIConfig:
    features: tuple[PSIFeature, ...]
    n_bins: int = 10
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", tuple(self.features))
        if not self.features:
            raise ValueError("features must not be empty")
        names = [feature.name for feature in self.features]
        if len(names) != len(set(names)):
            raise ValueError("Feature names must be unique")
        if isinstance(self.n_bins, bool) or not isinstance(self.n_bins, int):
            raise TypeError("n_bins must be an integer")
        if self.n_bins < 1:
            raise ValueError("n_bins must be a positive integer")
        if not np.isscalar(self.epsilon) or not 0 < float(self.epsilon) < 1:
            raise ValueError("epsilon must be between 0 and 1")


@dataclass(frozen=True)
class FeatureLevelConfig:
    name: str
    ftype: FeatureType
    levels: tuple[str, ...]
    boundaries: tuple[float, ...] = ()
    categories: tuple[Any, ...] = ()
    error_level: Any | None = None
    null_level: Any | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None


@dataclass(frozen=True)
class LevelConfig:
    features: tuple[FeatureLevelConfig, ...]
