"""Reusable training, evaluation, PSI and plots for ML projects."""

import logging as _logging
from typing import TYPE_CHECKING as _TYPE_CHECKING

if _TYPE_CHECKING:
    from .training.contracts import Pool
    from .training import TrainingResult, train_model

_logging.getLogger(__name__).addHandler(_logging.NullHandler())


__all__ = ["Pool", "TrainingResult", "train_model"]


def __getattr__(name: str) -> object:
    if name == "Pool":
        from .training.contracts import Pool

        return Pool
    if name == "TrainingResult":
        from .training import TrainingResult

        return TrainingResult
    if name == "train_model":
        from .training import train_model

        return train_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
