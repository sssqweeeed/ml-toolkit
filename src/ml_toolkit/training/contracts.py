"""Shared data contract and interfaces for training extensions."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import TYPE_CHECKING, Literal, Protocol

from ..data import Pool as Pool

if TYPE_CHECKING:
    from ..models.base import Model


class FeatureRanker(Protocol):
    """Describe a callable ranking a fitted adapter's current input columns.

    Notes
    -----
    Higher values indicate greater importance. Implementations may inspect the
    adapter's native_model, but must preserve its state and all input metadata.
    Explicit inheritance from this protocol is unnecessary.
    """

    def __call__(self, model: "Model", data: "Pool") -> Mapping[str, float]:
        """Return finite ranking values keyed by every current feature name.

        Parameters
        ----------
        model : Model
            Fitted adapter for the current feature subset.
        data : Pool
            Supplied validation rows with aligned features and weights.

        Returns
        -------
        Mapping[str, float]
            Exactly one finite numeric importance per data.X column. Signed
            values are allowed; larger values rank higher.
        """
        ...


class DatasetSplit(StrEnum):
    """Identify an evaluated dataset split.

    Attributes
    ----------
    TRAIN : DatasetSplit
        Rows used to fit the model; serialized as "train".
    VAL : DatasetSplit
        Validation rows; serialized as "val".
    TEST : DatasetSplit
        Independent test rows; serialized as "test". The training workflow
        evaluates test after final fitting when the splitter supplies held-out data.
    """

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


TrainingStage = Literal["tuning", "validation", "final"]
