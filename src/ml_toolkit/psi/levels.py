from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np
import polars as pl

from .config import (
    ERROR_LEVEL,
    NULL_LEVEL,
    UNSEEN_LEVEL,
    FeatureLevelConfig,
    LevelConfig,
    PSIFeature,
)


@dataclass
class LevelTransformer:
    """Learn reference categories and quantile boundaries for PSI.

    Parameters
    ----------
    features : tuple[PSIFeature, ...]
        Feature definitions with unique names.
    n_bins : int, default 10
        Positive requested number of numerical intervals.

    Attributes
    ----------
    config : LevelConfig or None
        Fitted level definitions; None until fit is called.

    Notes
    -----
    Numerical ties can reduce the number of quantile boundaries. Nulls,
    floating NaNs, and configured missing sentinels map to __NULL__.
    Categorical values absent from the reference map to __UNSEEN__; configured
    error sentinels and out-of-bound numerical values map to __ERROR__.
    Transformation retains unconfigured columns.
    """

    features: Sequence[PSIFeature]
    n_bins: int = 10
    config: LevelConfig | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.features = tuple(self.features)
        names = [feature.name for feature in self.features]
        if len(names) != len(set(names)):
            raise ValueError("Feature names must be unique")
        if self.n_bins < 1:
            raise ValueError("n_bins must be a positive integer")

    def fit(self, reference: pl.DataFrame) -> LevelTransformer:
        """Learn levels and numerical boundaries from reference data.

        Parameters
        ----------
        reference : polars.DataFrame
            Nonempty frame containing every configured feature. Regular numerical
            values must be finite and castable to Float64.

        Returns
        -------
        Self
            Transformer with fitted config; calling fit again replaces previous
            levels.

        Raises
        ------
        ValueError
            Reference data is empty, features are missing, regular numerical values
            are non-finite, or categories use reserved level names.
        """
        names = [feature.name for feature in self.features]
        missing = sorted(set(names) - set(reference.columns))
        if missing:
            raise ValueError(f"Features are missing from reference: {missing}")

        configs: list[FeatureLevelConfig] = []
        for feature in self.features:
            column = reference.get_column(feature.name)
            valid = column.is_not_null()
            if column.dtype.is_float():
                valid &= column.is_not_nan()
            if feature.null_level is not None:
                valid &= column != feature.null_level
            if feature.error_level is not None:
                valid &= column != feature.error_level
            regular = column.filter(valid)

            if feature.ftype == "categorical":
                regular = regular.unique(maintain_order=True)
                categories = tuple(regular.to_list())
                labels = tuple(regular.cast(pl.String).to_list())
                reserved = {NULL_LEVEL, ERROR_LEVEL, UNSEEN_LEVEL}
                if reserved.intersection(labels):
                    raise ValueError(
                        f"Feature {feature.name!r} contains a reserved PSI level"
                    )

                levels = (*labels, NULL_LEVEL, UNSEEN_LEVEL)
                if feature.error_level is not None:
                    levels = (*levels, ERROR_LEVEL)
                feature_config = FeatureLevelConfig(
                    name=feature.name,
                    ftype=feature.ftype,
                    levels=levels,
                    categories=categories,
                    error_level=feature.error_level,
                    null_level=feature.null_level,
                )
            else:
                numbers = regular.cast(pl.Float64, strict=True)
                if not numbers.is_finite().all():
                    raise ValueError(
                        f"Feature {feature.name!r} must contain finite values"
                    )
                if feature.lower_bound is not None:
                    numbers = numbers.filter(numbers >= feature.lower_bound)
                if feature.upper_bound is not None:
                    numbers = numbers.filter(numbers <= feature.upper_bound)
                minimum = numbers.min()
                maximum = numbers.max()
                if len(numbers) < 2 or minimum == maximum:
                    boundaries: tuple[float, ...] = ()
                else:
                    quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]
                    boundaries = tuple(
                        float(value)
                        for value in np.unique(
                            np.quantile(numbers.to_numpy(), quantiles)
                        )
                        if minimum < value < maximum
                    )

                lower_edge = (
                    -np.inf if feature.lower_bound is None else feature.lower_bound
                )
                upper_edge = (
                    np.inf if feature.upper_bound is None else feature.upper_bound
                )
                interval_edges = (lower_edge, *boundaries, upper_edge)
                interval_levels: list[str] = []
                for left, right in pairwise(interval_edges):
                    left_label = "-inf" if left == -np.inf else f"{left:g}"
                    right_label = "inf" if right == np.inf else f"{right:g}"
                    right_bracket = (
                        "]" if right == upper_edge and right != np.inf else ")"
                    )
                    interval_levels.append(
                        f"[{left_label}, {right_label}{right_bracket}"
                    )

                levels = (*interval_levels, NULL_LEVEL)
                if (
                    feature.error_level is not None
                    or feature.lower_bound not in (None, -np.inf)
                    or feature.upper_bound not in (None, np.inf)
                ):
                    levels = (*levels, ERROR_LEVEL)
                feature_config = FeatureLevelConfig(
                    name=feature.name,
                    ftype=feature.ftype,
                    levels=levels,
                    boundaries=boundaries,
                    error_level=feature.error_level,
                    null_level=feature.null_level,
                    lower_bound=feature.lower_bound,
                    upper_bound=feature.upper_bound,
                )
            configs.append(feature_config)

        self.config = LevelConfig(tuple(configs))
        return self

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        """Replace configured features with fitted categorical level labels.

        Parameters
        ----------
        data : polars.DataFrame
            Nonempty frame containing the configured features.

        Returns
        -------
        polars.DataFrame
            Frame with configured columns cast to Categorical; unconfigured columns
            and row order are retained.

        Raises
        ------
        RuntimeError
            The transformer has not been fitted.
        ValueError
            Data is empty or configured features are missing.

        Notes
        -----
        Numerical intervals are left-inclusive. A value equal to an internal
        boundary enters the interval on its right.
        """
        if self.config is None:
            raise RuntimeError("LevelTransformer must be fitted before transform")
        missing = sorted(
            {feature.name for feature in self.config.features} - set(data.columns)
        )
        if missing:
            raise ValueError(f"Features are missing from data: {missing}")

        expressions: list[pl.Expr] = []
        for feature in self.config.features:
            column = pl.col(feature.name)
            nulls = column.is_null()
            if data.schema[feature.name].is_float():
                nulls |= column.is_nan()
            if feature.null_level is not None:
                nulls |= column == feature.null_level

            errors = pl.lit(False)
            if feature.error_level is not None:
                errors = column == feature.error_level

            if feature.ftype == "categorical":
                regular = (
                    pl.when(column.is_in(feature.categories))
                    .then(column.cast(pl.String))
                    .otherwise(pl.lit(UNSEEN_LEVEL))
                )
            else:
                special_errors = errors
                numbers = (
                    pl.when(nulls | special_errors)
                    .then(pl.lit(None))
                    .otherwise(column)
                    .cast(pl.Float64, strict=True)
                )
                if feature.lower_bound is not None:
                    errors |= (numbers < feature.lower_bound).fill_null(False)
                if feature.upper_bound is not None:
                    errors |= (numbers > feature.upper_bound).fill_null(False)
                interval_levels = feature.levels[: len(feature.boundaries) + 1]
                regular = pl.lit(interval_levels[-1])
                for index in reversed(range(len(feature.boundaries))):
                    regular = (
                        pl.when(numbers < feature.boundaries[index])
                        .then(pl.lit(interval_levels[index]))
                        .otherwise(regular)
                    )

            expression = (
                pl.when(nulls)
                .then(pl.lit(NULL_LEVEL))
                .when(errors)
                .then(pl.lit(ERROR_LEVEL))
                .otherwise(regular)
                .cast(pl.Categorical)
                .alias(feature.name)
            )
            expressions.append(expression)

        return data.with_columns(expressions)
