from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import cast

import numpy as np
import polars as pl

from .config import (
    ERROR_LEVEL,
    NULL_LEVEL,
    UNSEEN_LEVEL,
    FeatureLevels,
    FeatureLevelConfig,
    FeatureType,
    FeatureScalar,
)
from ._serialization import decode_scalar, encode_scalar, mapping


def _interval_labels(
    lower_edge: float, boundaries: tuple[float, ...], upper_edge: float
) -> list[str]:
    edges = (lower_edge, *boundaries, upper_edge)
    finite = [edge for edge in edges if np.isfinite(edge)]
    # Use the shortest precision that keeps every finite edge distinguishable.
    precision = next(
        (
            p
            for p in range(6, 17)
            if len({f"{edge:.{p}g}" for edge in finite}) == len(finite)
        ),
        17,
    )

    def text(edge: float) -> str:
        if edge == -np.inf:
            return "-inf"
        if edge == np.inf:
            return "inf"
        return f"{edge:.{precision}g}"

    return [
        f"[{text(left)}, {text(right)}"
        f"{']' if right == upper_edge and right != np.inf else ')'}"
        for left, right in pairwise(edges)
    ]


def _legacy_interval_labels(
    lower_edge: float, boundaries: tuple[float, ...], upper_edge: float
) -> list[str]:
    # Pre-uniqueness format, accepted when loading old states.
    return [
        f"[{'-inf' if a == -np.inf else f'{a:g}'}, "
        f"{'inf' if b == np.inf else f'{b:g}'}"
        f"{']' if b == upper_edge and b != np.inf else ')'}"
        for a, b in pairwise((lower_edge, *boundaries, upper_edge))
    ]


@dataclass
class LevelTransformer:
    """Learn reference categories and quantile boundaries for diagnostics.

    Parameters
    ----------
    features : sequence of FeatureLevelConfig
        Feature definitions with unique names; stored as a tuple.
    n_bins : int, default 10
        Positive requested number of numerical intervals.

    Attributes
    ----------
    feature_levels : tuple[FeatureLevels, ...] or None
        Levels in feature order; None until fit is called.

    Notes
    -----
    Numerical ties can reduce the number of quantile boundaries. Nulls,
    floating NaNs, and configured missing sentinels map to __NULL__.
    Categorical values absent from the reference map to __UNSEEN__; configured
    error sentinels and out-of-bound numerical values map to __ERROR__.
    Transformation retains unconfigured columns.
    """

    features: Sequence[FeatureLevelConfig]
    n_bins: int = 10
    feature_levels: tuple[FeatureLevels, ...] | None = field(init=False, default=None)

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
            Frame containing every configured feature. Regular numerical
            values must be castable to Float64; infinities are valid.

        Returns
        -------
        Self
            Transformer with reference-based levels. Calling fit again replaces
            its levels; previously returned tuples retain their original values.

        Raises
        ------
        ValueError
            Features are missing or categories use reserved level names.

        Notes
        -----
        An empty frame must still supply typed feature columns. Features with
        no regular reference values learn no categories or numerical cut points.
        Infinities are valid values: cut points use finite values only.
        """
        names = [feature.name for feature in self.features]
        missing = sorted(set(names) - set(reference.columns))
        if missing:
            raise ValueError(f"Features are missing from reference: {missing}")

        feature_levels: list[FeatureLevels] = []
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
                        f"Feature {feature.name!r} contains a reserved feature level"
                    )

                levels = (*labels, NULL_LEVEL, UNSEEN_LEVEL)
                if feature.error_level is not None:
                    levels = (*levels, ERROR_LEVEL)
                result = FeatureLevels(
                    config=feature,
                    levels=levels,
                    categories=categories,
                )
            else:
                numbers = regular.cast(pl.Float64, strict=True)
                if feature.lower_bound is not None:
                    numbers = numbers.filter(numbers >= feature.lower_bound)
                if feature.upper_bound is not None:
                    numbers = numbers.filter(numbers <= feature.upper_bound)
                # Infinities are valid values but cannot define cut points.
                numbers = numbers.filter(numbers.is_finite())
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
                interval_levels = _interval_labels(lower_edge, boundaries, upper_edge)

                levels = (*interval_levels, NULL_LEVEL)
                if (
                    feature.error_level is not None
                    or feature.lower_bound not in (None, -np.inf)
                    or feature.upper_bound not in (None, np.inf)
                ):
                    levels = (*levels, ERROR_LEVEL)
                result = FeatureLevels(
                    config=feature,
                    levels=levels,
                    boundaries=boundaries,
                )
            feature_levels.append(result)

        self.feature_levels = tuple(feature_levels)
        return self

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        """Replace configured features with fitted categorical level labels.

        Parameters
        ----------
        data : polars.DataFrame
            Frame containing the configured features; may have zero rows.

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
            Configured features are missing.

        Notes
        -----
        Numerical intervals are left-inclusive. A value equal to an internal
        boundary enters the interval on its right. Negative infinity enters
        the first interval and positive infinity the last, unless configured
        bounds exclude them, in which case they map to __ERROR__.
        """
        if self.feature_levels is None:
            raise RuntimeError("LevelTransformer must be fitted before transform")
        missing = sorted(
            {feature.config.name for feature in self.feature_levels} - set(data.columns)
        )
        if missing:
            raise ValueError(f"Features are missing from data: {missing}")

        expressions: list[pl.Expr] = []
        for feature in self.feature_levels:
            column = pl.col(feature.config.name)
            nulls = column.is_null()
            if data.schema[feature.config.name].is_float():
                nulls |= column.is_nan()
            if feature.config.null_level is not None:
                nulls |= column == feature.config.null_level

            errors = pl.lit(False)
            if feature.config.error_level is not None:
                errors = column == feature.config.error_level

            if feature.config.ftype == "categorical":
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
                if feature.config.lower_bound is not None:
                    errors |= (numbers < feature.config.lower_bound).fill_null(False)
                if feature.config.upper_bound is not None:
                    errors |= (numbers > feature.config.upper_bound).fill_null(False)
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
                .alias(feature.config.name)
            )
            expressions.append(expression)

        return data.with_columns(expressions)

    def to_dict(self) -> dict[str, object]:
        """Serialize fitted state without rounding boundaries or losing scalar types."""
        if self.feature_levels is None:
            raise RuntimeError("LevelTransformer must be fitted before serialization")
        return {
            "version": 1,
            "n_bins": self.n_bins,
            "features": [
                {
                    "name": feature.config.name,
                    "ftype": feature.config.ftype,
                    "null_level": encode_scalar(feature.config.null_level),
                    "error_level": encode_scalar(feature.config.error_level),
                    "lower_bound": encode_scalar(feature.config.lower_bound),
                    "upper_bound": encode_scalar(feature.config.upper_bound),
                    "levels": list(feature.levels),
                    "boundaries": [encode_scalar(x) for x in feature.boundaries],
                    "categories": [encode_scalar(x) for x in feature.categories],
                }
                for feature in self.feature_levels
            ],
        }

    @classmethod
    def from_dict(cls, payload: object) -> LevelTransformer:
        """Restore validated fitted state; never refit on new observations."""
        try:
            data = mapping(payload)
            if type(data["version"]) is not int or data["version"] != 1:
                raise ValueError("Unsupported LevelTransformer format version")
            if type(data["n_bins"]) is not int or data["n_bins"] < 1:
                raise ValueError("Invalid n_bins")
            if not isinstance(data["features"], list):
                raise ValueError("Invalid feature definitions")
            fitted: list[FeatureLevels] = []
            for raw in cast(list[object], data["features"]):
                item = mapping(raw)
                if not isinstance(item["name"], str):
                    raise ValueError("Invalid feature name")
                lower = decode_scalar(item["lower_bound"])
                upper = decode_scalar(item["upper_bound"])
                for bound in (lower, upper):
                    if bound is not None and type(bound) not in (float, int):
                        raise ValueError("Invalid numerical bound")
                config = FeatureLevelConfig(
                    name=item["name"],
                    ftype=cast(FeatureType, item["ftype"]),
                    null_level=decode_scalar(item["null_level"]),
                    error_level=decode_scalar(item["error_level"]),
                    lower_bound=cast(float | None, lower),
                    upper_bound=cast(float | None, upper),
                )
                for key in ("levels", "boundaries", "categories"):
                    if not isinstance(item[key], list):
                        raise ValueError(f"Invalid {key}")
                labels = tuple(item["levels"])
                if not labels or any(not isinstance(x, str) for x in labels):
                    raise ValueError("Invalid level labels")
                cuts = tuple(decode_scalar(x) for x in item["boundaries"])
                if any(type(x) is not float or not np.isfinite(x) for x in cuts):
                    raise ValueError("Invalid numerical boundaries")
                boundaries = cast(tuple[float, ...], cuts)
                categories = tuple(decode_scalar(x) for x in item["categories"])
                if any(x is None for x in categories):
                    raise ValueError("Null is not a regular category")
                if config.ftype == "numerical":
                    left = -np.inf if lower is None else cast(float, lower)
                    right = np.inf if upper is None else cast(float, upper)
                    if (
                        categories
                        or len(boundaries) >= data["n_bins"]
                        or any(not left < x < right for x in boundaries)
                        or any(a >= b for a, b in pairwise(boundaries))
                    ):
                        raise ValueError("Inconsistent numerical levels")
                    unique_labels = _interval_labels(left, boundaries, right)
                    legacy_labels = _legacy_interval_labels(left, boundaries, right)
                    suffix = [NULL_LEVEL]
                    if (
                        config.error_level is not None
                        or left != -np.inf
                        or right != np.inf
                    ):
                        suffix.append(ERROR_LEVEL)
                    expected = unique_labels + suffix
                    if list(labels) not in (
                        expected,
                        legacy_labels + suffix,
                    ):
                        raise ValueError("Level labels do not match fitted definitions")
                    labels = tuple(expected)
                else:
                    if boundaries:
                        raise ValueError("Categorical features cannot have boundaries")
                    # Labels retain the source dtype's display format (e.g.
                    # Float32 or millisecond timestamps). Inferring a fresh
                    # Polars dtype from Python scalars can change that format.
                    expected = list(labels[: len(categories)])
                    if (
                        len(set(categories)) != len(categories)
                        or any(isinstance(x, float) and np.isnan(x) for x in categories)
                        or len(set(expected)) != len(expected)
                        or set(expected).intersection(
                            (NULL_LEVEL, ERROR_LEVEL, UNSEEN_LEVEL)
                        )
                    ):
                        raise ValueError("Invalid reference categories")
                    expected += [NULL_LEVEL, UNSEEN_LEVEL]
                    if config.error_level is not None:
                        expected.append(ERROR_LEVEL)
                if list(labels) != expected:
                    raise ValueError("Level labels do not match fitted definitions")
                fitted.append(
                    FeatureLevels(
                        config,
                        labels,
                        boundaries,
                        cast(tuple["FeatureScalar", ...], categories),
                    )
                )
            result = cls(tuple(f.config for f in fitted), n_bins=data["n_bins"])
            result.feature_levels = tuple(fitted)
            return result
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid LevelTransformer state: {exc}") from exc
