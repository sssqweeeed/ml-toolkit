from collections import Counter

import numpy as np
import polars as pl


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
        Nonempty Categorical Series representing the reference levels.
    current : polars.Series
        Nonempty Categorical Series representing the current levels; sample
        sizes may differ.
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
    TypeError
        Either input is not a Categorical Polars Series.
    ValueError
        Either sample is empty or epsilon is invalid.

    Notes
    -----
    Shares are floored independently without renormalization. Use
    LevelTransformer or PSIAnalysis to bin raw numerical values first.

    Examples
    --------
    >>> import polars as pl
    >>> from ml_toolkit.psi import psi
    >>> reference = pl.Series(["a", "b"], dtype=pl.Categorical)
    >>> current = pl.Series(["b", "a"], dtype=pl.Categorical)
    >>> psi(reference, current)
    0.0
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
    reference_share = np.maximum(reference_share, epsilon)
    current_share = np.maximum(current_share, epsilon)

    return float(
        np.sum(
            (current_share - reference_share) * np.log(current_share / reference_share)
        )
    )
