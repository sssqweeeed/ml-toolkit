"""Shared feature definitions and fitted diagnostic levels."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

import numpy as np

FeatureScalar = str | int | float | bool | date | datetime


FeatureType = Literal["categorical", "numerical"]

NULL_LEVEL = "__NULL__"
ERROR_LEVEL = "__ERROR__"
UNSEEN_LEVEL = "__UNSEEN__"


@dataclass(frozen=True)
class FeatureLevelSpec:
    """Configure how one feature is assigned to diagnostic levels.

    Parameters
    ----------
    name : str
        Nonempty feature column name.
    ftype : {'categorical', 'numerical'}
        Choose category levels or reference quantile intervals.
    error_level : FeatureScalar or None, default None
        Optional sentinel mapped to __ERROR__; must differ from null_level and
        must not be NaN.
    null_level : FeatureScalar or None, default None
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
    reserved diagnostic level strings.

    Examples
    --------
    >>> from ml_toolkit.feature_levels import FeatureLevelSpec
    >>> feature = FeatureLevelSpec("age", "numerical", lower_bound=0)
    >>> feature.name, feature.lower_bound
    ('age', 0)
    """

    name: str
    ftype: FeatureType
    error_level: FeatureScalar | None = None
    null_level: FeatureScalar | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Feature name must not be empty")
        if self.ftype not in ("categorical", "numerical"):
            raise ValueError(f"Unknown feature type: {self.ftype!r}")
        if isinstance(self.error_level, float) and np.isnan(self.error_level):
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
class FeatureLevelConfig:
    """Store learned levels for one feature.

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
    categories : tuple[FeatureScalar, ...]
        Reference categories in first-occurrence order; empty for numerical
        features.
    error_level : FeatureScalar or None
        Configured error sentinel.
    null_level : FeatureScalar or None
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
    categories: tuple[FeatureScalar, ...] = ()
    error_level: FeatureScalar | None = None
    null_level: FeatureScalar | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None


@dataclass(frozen=True)
class LevelConfig:
    """Store fitted diagnostic level definitions in feature order.

    Attributes
    ----------
    features : tuple[FeatureLevelConfig, ...]
        Learned levels and transformation settings for each feature.
    """

    features: tuple[FeatureLevelConfig, ...]
