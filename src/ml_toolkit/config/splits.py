"""Chronological holdout and time-series validation settings."""

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True, slots=True)
class HoldoutConfig:
    """Reserve the trailing rows of a chronological training dataset.

    Parameters
    ----------
    val_size : int or float, default 0.2
        Integer row count or fraction in (0, 1), rounded up to whole rows.
    gap : int, default 0
        Nonnegative rows omitted immediately before validation.
    """

    val_size: int | float = 0.2
    gap: int = 0

    def __post_init__(self) -> None:
        if self.val_size <= 0 or (
            isinstance(self.val_size, float) and not 0 < self.val_size < 1
        ):
            raise ValueError(
                "val_size must be positive rows or a fraction between 0 and 1"
            )
        if self.gap < 0:
            raise ValueError("gap must be nonnegative")


@dataclass(frozen=True, kw_only=True, slots=True)
class TimeSeriesCVConfig:
    """Repeat chronological train/validation splits with expanding training.

    Parameters
    ----------
    n_splits : int
        Number of generated splits, at least two.
    val_size : int, float, or None, default None
        Validation rows or fraction of the complete input, rounded up.
        None uses TimeSeriesSplit's equal-block default.
    gap : int, default 0
        Nonnegative row gap before each validation block.
    max_train_size : int or None, default None
        Maximum training rows; None includes all preceding rows.
    skip_splits : int, default 0
        Initial splits to omit. Must be smaller than n_splits.
    """

    n_splits: int
    val_size: int | float | None = None
    gap: int = 0
    max_train_size: int | None = None
    skip_splits: int = 0

    def __post_init__(self) -> None:
        if self.val_size is not None:
            HoldoutConfig(val_size=self.val_size, gap=self.gap)
        elif self.gap < 0:
            raise ValueError("gap must be nonnegative")
        if self.max_train_size is not None and self.max_train_size < 1:
            raise ValueError("max_train_size must be positive")
        if self.skip_splits < 0:
            raise ValueError("skip_splits must be nonnegative")
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if self.skip_splits >= self.n_splits:
            raise ValueError("skip_splits must be smaller than n_splits")
