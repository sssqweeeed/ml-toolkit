"""PSI scoring kernel, configuration and result types."""

from collections import Counter
from dataclasses import dataclass

import numpy as np
import polars as pl

from .._time import PERIOD_INTERVALS
from ..feature_levels import FeatureLevelConfig, FeatureLevels
from ..types import FloatArray, TimePeriod


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


@dataclass(frozen=True)
class PSIResult:
    """Store PSI scores and distribution tables as Polars DataFrames.

    Attributes
    ----------
    psi : polars.DataFrame
        One row per feature with feature (String) and psi (Float64) columns.
    level_distribution : polars.DataFrame
        Columns feature, sample, and level (String), rows (Int64), and share
        (Float64). Samples are reference and current; every fitted level is
        present, including zero-share levels.
    time_distribution : polars.DataFrame
        Columns feature, sample, and level (String), period_start (Datetime), and share
        (Float64). Contains every fitted level in each observed sample period;
        preserves the input timezone. Empty with naive Datetime when time was omitted.
    feature_levels : tuple[FeatureLevels, ...]
        Reference-fitted categories, interval boundaries, and special level
        definitions.
    """

    psi: pl.DataFrame
    level_distribution: pl.DataFrame
    time_distribution: pl.DataFrame
    feature_levels: tuple[FeatureLevels, ...]


def psi_from_shares(
    reference_share: FloatArray, current_share: FloatArray, *, epsilon: float
) -> float:
    """Shared scoring kernel for row-based and persisted aggregate references."""
    reference_share = np.maximum(reference_share, epsilon)
    current_share = np.maximum(current_share, epsilon)
    return float(
        np.sum(
            (current_share - reference_share) * np.log(current_share / reference_share)
        )
    )


def psi(
    reference: pl.Series,
    current: pl.Series,
    *,
    epsilon: float = 1e-6,
) -> float:
    """Calculate PSI for two categorical feature distributions.

    Parameters
    ----------
    reference : polars.Series
        Nonempty Series of discrete reference levels, compared after casting
        to String. Bin continuous numerical values before calling.
    current : polars.Series
        Nonempty Series of discrete current levels, compared after casting to
        String; sample sizes may differ. Null is counted as a separate level.
    epsilon : float, default 1e-6
        Minimum share strictly between zero and one used to avoid logarithms of
        zero.

    Returns
    -------
    float
        Sum of (current_share - reference_share) * log(current_share /
        reference_share) over the union of observed levels.

    Raises
    ------
    ValueError
        Either sample is empty or epsilon is invalid.

    Notes
    -----
    Shares are floored independently without renormalization. Use
    LevelTransformer or PSIAnalysis to bin raw numerical values first.

    Examples
    --------
    Compare category shares: north falls from 75% to 25%, while south rises.

    >>> import polars as pl
    >>> from ml_toolkit.psi import psi
    >>> reference = pl.Series(["north"] * 3 + ["south"], dtype=pl.Categorical)
    >>> current = pl.Series(["north"] + ["south"] * 3, dtype=pl.Categorical)
    >>> round(psi(reference, current), 4)
    1.0986
    """
    if reference.is_empty() or current.is_empty():
        raise ValueError("PSI input series must not be empty")
    if not 0 < epsilon < 1:
        raise ValueError("epsilon must be between 0 and 1")

    reference_counts = Counter(reference.cast(pl.String).to_list())
    current_counts = Counter(current.cast(pl.String).to_list())
    levels = reference_counts.keys() | current_counts.keys()

    reference_share = np.asarray(
        [reference_counts[level] / len(reference) for level in levels], dtype=float
    )
    current_share = np.asarray(
        [current_counts[level] / len(current) for level in levels], dtype=float
    )
    return psi_from_shares(reference_share, current_share, epsilon=epsilon)
