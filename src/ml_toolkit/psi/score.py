from collections import Counter

import numpy as np
import polars as pl


def psi(
    reference: pl.Series,
    current: pl.Series,
    *,
    epsilon: float = 1e-6,
) -> float:
    """Calculate PSI for two categorical feature distributions."""
    if not isinstance(reference, pl.Series) or reference.dtype != pl.Categorical:
        raise TypeError("reference must be a pl.Categorical Series")
    if not isinstance(current, pl.Series) or current.dtype != pl.Categorical:
        raise TypeError("current must be a pl.Categorical Series")
    if reference.is_empty() or current.is_empty():
        raise ValueError("PSI input series must not be empty")
    if not np.isscalar(epsilon) or not 0 < float(epsilon) < 1:
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
    reference_share = np.maximum(reference_share, epsilon)
    current_share = np.maximum(current_share, epsilon)

    return float(
        np.sum(
            (current_share - reference_share) * np.log(current_share / reference_share)
        )
    )
