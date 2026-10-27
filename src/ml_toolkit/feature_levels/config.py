"""Shared feature definitions and fitted diagnostic levels."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from ..data import Pool

FeatureScalar = str | int | float | bool | date | datetime


FeatureType = Literal["categorical", "numerical"]

NULL_LEVEL = "__NULL__"
ERROR_LEVEL = "__ERROR__"
UNSEEN_LEVEL = "__UNSEEN__"


@dataclass(frozen=True)
class FeatureLevelConfig:
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
    Fit age intervals once, then keep invalid and missing values separate.

    >>> import polars as pl
    >>> from ml_toolkit.feature_levels import FeatureLevelConfig, LevelTransformer
    >>> feature = FeatureLevelConfig("age", "numerical", lower_bound=0)
    >>> levels = LevelTransformer((feature,), n_bins=2).fit(
    ...     pl.DataFrame({"age": [20, 30, 40, 50]})
    ... )
    >>> current = pl.DataFrame({"age": [-1, None, 25, 45]})
    >>> levels.transform(current)["age"].cast(pl.String).to_list()
    ['__ERROR__', '__NULL__', '[0, 35)', '[35, inf)']
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
class FeatureLevels:
    """Store categories and intervals determined from reference data.

    Attributes
    ----------
    config : FeatureLevelConfig
        Immutable rules used to determine and apply these levels.
    levels : tuple[str, ...]
        Ordered interval or category labels followed by applicable special
        levels.
    boundaries : tuple[float, ...]
        Internal numerical cut points; empty for categorical or single-interval
        features.
    categories : tuple[FeatureScalar, ...]
        Reference categories in first-occurrence order; empty for numerical
        features.
    """

    config: FeatureLevelConfig
    levels: tuple[str, ...]
    boundaries: tuple[float, ...] = ()
    categories: tuple[FeatureScalar, ...] = ()


def feature_definitions(
    data: Pool, features: tuple[FeatureLevelConfig, ...] | None
) -> tuple[FeatureLevelConfig, ...]:
    if features is not None:
        return features
    return tuple(
        FeatureLevelConfig(
            name,
            (
                "categorical"
                if name in data.cat_features or not dtype.is_numeric()
                else "numerical"
            ),
        )
        for name, dtype in data.X.schema.items()
    )


def validate_features(features: tuple[str, ...] | None) -> None:
    if features is not None and (
        any(not name.strip() for name in features)
        or len(set(features)) != len(features)
    ):
        raise ValueError("features must contain unique nonempty names")
