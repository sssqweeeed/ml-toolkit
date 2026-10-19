"""Fit PSI levels once and compare any Polars feature tables."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import polars as pl

from .config import PSIConfig
from .distributions import (
    MONTHLY_SCHEMA,
    level_distributions,
    monthly_level_distributions,
)
from ..feature_levels import LevelConfig, LevelTransformer
from .results import PSIResult
from .score import psi

logger = logging.getLogger(__name__)


def _resolve_time(data: pl.DataFrame, time: pl.Series | str | None) -> pl.Series | None:
    if time is None:
        return None
    if isinstance(time, str):
        time = data.get_column(time)
    if len(time) != len(data):
        raise ValueError("time must align with data rows")
    if time.null_count():
        raise ValueError("time must not contain null values")
    if time.dtype != pl.Date and time.dtype.base_type() != pl.Datetime:
        raise TypeError("time must have Date or Datetime dtype")
    return time


@dataclass
class PSIAnalysis:
    """Compare a current population against fitted reference feature levels.

    Parameters
    ----------
    config : PSIConfig
        Feature definitions, numerical bin count, and share floor.

    Attributes
    ----------
    transformer : LevelTransformer
        Transformer learning levels from reference data.
    reference_levels : polars.DataFrame or None
        Transformed reference frame; None before fitting.
    reference_time : polars.Series or None
        Optional reference timestamps for monthly distributions.

    Examples
    --------
    >>> import polars as pl
    >>> from ml_toolkit.feature_levels import FeatureLevelSpec
    >>> from ml_toolkit.psi import PSIAnalysis, PSIConfig
    >>> reference = pl.DataFrame({"segment": ["a", "a", "b", "b"]})
    >>> current = pl.DataFrame({"segment": ["a", "b", "a", "b"]})
    >>> analysis = PSIAnalysis(PSIConfig((FeatureLevelSpec("segment", "categorical"),)))
    >>> result = analysis.fit(reference).compare(current)
    >>> result.psi.get_column("psi").to_list()
    [0.0]
    """

    config: PSIConfig
    transformer: LevelTransformer = field(init=False)
    reference_levels: pl.DataFrame | None = field(init=False, default=None)
    reference_time: pl.Series | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.transformer = LevelTransformer(
            self.config.features, n_bins=self.config.n_bins
        )

    def fit(
        self,
        reference: pl.DataFrame,
        *,
        time: pl.Series | str | None = None,
    ) -> PSIAnalysis:
        """Learn feature levels from reference data and optionally record dates.

        Parameters
        ----------
        reference : polars.DataFrame
            Reference frame containing all configured features. An empty typed
            frame can be fitted, but comparison requires a nonempty reference.
        time : polars.Series, str, or None, default None
            Optional non-null Date/Datetime Series aligned with rows, or the name
            of a timestamp column. Provide time for both samples or neither.

        Returns
        -------
        Self
            Analysis with stored reference levels and optional timestamps.

        Raises
        ------
        ValueError
            Features are missing, or supplied timestamps
            are null or misaligned.
        TypeError
            Supplied timestamps are not Date or Datetime.
        """
        reference_time = _resolve_time(reference, time)
        logger.debug(
            "Fitting PSI reference: %s rows, %s features",
            len(reference),
            len(self.config.features),
        )
        self.transformer.fit(reference)
        self.reference_levels = self.transformer.transform(reference)
        self.reference_time = reference_time
        logger.debug("Fitted PSI reference levels")
        return self

    def compare(
        self,
        current: pl.DataFrame,
        *,
        time: pl.Series | str | None = None,
    ) -> PSIResult:
        """Score current data using the fitted reference levels.

        Parameters
        ----------
        current : polars.DataFrame
            Nonempty current frame containing all configured features.
        time : polars.Series, str, or None, default None
            Optional non-null Date/Datetime Series aligned with rows, or the name
            of a timestamp column. Provide time for both samples or neither.

        Returns
        -------
        PSIResult
            Per-feature PSI, level shares for both samples, optional monthly
            shares, and fitted level definitions.

        Raises
        ------
        RuntimeError
            The analysis has not been fitted.
        ValueError
            Data is empty, required features are missing, timestamps are invalid,
            or only one sample supplies time.
        """
        if self.reference_levels is None or self.transformer.config is None:
            raise RuntimeError("PSIAnalysis must be fitted before compare")
        current_time = _resolve_time(current, time)
        if (self.reference_time is None) != (current_time is None):
            raise ValueError("Both samples must provide time for monthly distributions")

        logger.debug(
            "Comparing PSI: %s current rows, %s features",
            len(current),
            len(self.config.features),
        )
        current_levels = self.transformer.transform(current)
        return compare_psi(
            self.reference_levels,
            current_levels,
            self.transformer.config,
            epsilon=self.config.epsilon,
            reference_time=self.reference_time,
            current_time=current_time,
        )


def compare_psi(
    reference_levels: pl.DataFrame,
    current_levels: pl.DataFrame,
    level_config: LevelConfig,
    *,
    epsilon: float = 1e-6,
    reference_time: pl.Series | None = None,
    current_time: pl.Series | None = None,
) -> PSIResult:
    """Compare populations already transformed using shared reference levels.

    Parameters
    ----------
    reference_levels, current_levels : polars.DataFrame
        Nonempty frames with the configured columns transformed by the same
        LevelTransformer. Extra columns are ignored.
    level_config : LevelConfig
        Fitted definitions, including the order of features and levels.
    epsilon : float, default 1e-6
        Share floor strictly between zero and one for PSI logarithms.
    reference_time, current_time : polars.Series or None, default None
        Aligned, non-null Date/Datetime values. Supply both to include monthly
        distributions, or neither.

    Returns
    -------
    PSIResult
        Per-feature scores, level distributions, optional monthly distributions,
        and the supplied fitted definitions. No levels are fitted here.

    Raises
    ------
    ValueError
        Frames or feature definitions are empty, configured columns or levels
        are missing or invalid, timestamps are misaligned, or epsilon is invalid.
    TypeError
        Feature columns are not Categorical or timestamps have an invalid dtype.
    """
    if not level_config.features:
        raise ValueError("Feature definitions must not be empty")
    if not 0 < epsilon < 1:
        raise ValueError("epsilon must be between 0 and 1")
    for frame in (reference_levels, current_levels):
        if frame.is_empty():
            raise ValueError("PSI frames must not be empty")
        missing = sorted(
            {feature.name for feature in level_config.features} - set(frame.columns)
        )
        if missing:
            raise ValueError(f"Features are missing from levels: {missing}")
        for feature in level_config.features:
            values = frame.get_column(feature.name)
            if values.dtype != pl.Categorical:
                raise TypeError("PSI level columns must be Categorical")
            if (
                values.null_count()
                or not values.cast(pl.String).is_in(feature.levels).all()
            ):
                raise ValueError(f"Feature {feature.name!r} contains undefined levels")
    reference_time = _resolve_time(reference_levels, reference_time)
    current_time = _resolve_time(current_levels, current_time)
    if (reference_time is None) != (current_time is None):
        raise ValueError("Both samples must provide time for monthly distributions")
    feature_scores = pl.DataFrame(
        {
            "feature": [feature.name for feature in level_config.features],
            "psi": [
                psi(
                    reference_levels.get_column(feature.name),
                    current_levels.get_column(feature.name),
                    epsilon=epsilon,
                )
                for feature in level_config.features
            ],
        },
        schema={"feature": pl.String, "psi": pl.Float64},
    )
    level_distribution = level_distributions(
        reference_levels, current_levels, level_config
    )
    monthly_distribution = pl.DataFrame(schema=MONTHLY_SCHEMA)
    if reference_time is not None and current_time is not None:
        monthly_distribution = monthly_level_distributions(
            reference_levels,
            reference_time,
            current_levels,
            current_time,
            level_config,
        )
    result = PSIResult(
        psi=feature_scores,
        level_distribution=level_distribution,
        monthly_distribution=monthly_distribution,
        level_config=level_config,
    )
    logger.debug("Completed PSI comparison: %s feature scores", len(feature_scores))
    return result
