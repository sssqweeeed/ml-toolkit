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
    """Configure how one feature is assigned to PSI levels.

    Parameters
    ----------
    name : str
        Nonempty feature column name.
    ftype : {'categorical', 'numerical'}
        Choose category levels or reference quantile intervals.
    error_level : Any or None, default None
        Optional sentinel mapped to __ERROR__; must differ from null_level and
        must not be NaN.
    null_level : Any or None, default None
        Optional sentinel treated as missing alongside actual nulls and
        floating NaNs.
    lower_bound : float or None, default None
        Inclusive numerical lower bound; values below it map to __ERROR__. None
        leaves the lower tail unbounded.
    upper_bound : float or None, default None
        Inclusive numerical upper bound; values above it map to __ERROR__. None
        leaves the upper tail unbounded.

    Notes
    -----
    Bounds apply only to numerical features. Regular categories must not use
    reserved PSI level strings.

    Examples
    --------
    >>> from ml_toolkit.psi import PSIFeature
    >>> feature = PSIFeature("age", "numerical", lower_bound=0)
    >>> feature.name, feature.lower_bound
    ('age', 0)
    """

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
    """Configure reference-based population stability analysis.

    Parameters
    ----------
    features : tuple[PSIFeature, ...]
        Nonempty collection of uniquely named feature definitions, normalized
        to a tuple.
    n_bins : int, default 10
        Positive requested number of numerical quantile bins. Ties and constant
        features may produce fewer bins.
    epsilon : float, default 1e-6
        Share floor strictly between zero and one, applied before logarithms.
    """

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
    """Store learned levels for one PSI feature.

    Attributes
    ----------
    name : str
        Original feature name.
    ftype : {'categorical', 'numerical'}
        Feature interpretation used during transformation.
    levels : tuple[str, ...]
        Ordered interval or category labels followed by applicable special
        levels.
    boundaries : tuple[float, ...]
        Internal numerical cut points; empty for categorical or single-interval
        features.
    categories : tuple[Any, ...]
        Reference categories in first-occurrence order; empty for numerical
        features.
    error_level : Any or None
        Configured error sentinel.
    null_level : Any or None
        Configured missing-value sentinel.
    lower_bound : float or None
        Inclusive numerical lower bound.
    upper_bound : float or None
        Inclusive numerical upper bound.
    """

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
    """Store fitted PSI level definitions in feature order.

    Attributes
    ----------
    features : tuple[FeatureLevelConfig, ...]
        Learned levels and transformation settings for each feature.
    """

    features: tuple[FeatureLevelConfig, ...]
