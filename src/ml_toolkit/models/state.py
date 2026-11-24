"""Schemas and loading checks for persisted native-model serving state."""

import json
from pathlib import Path
from typing import Any, NotRequired, TypedDict, cast

from ..inflation import InflationConfig
from ..types import ModelParams


class RegressionState(TypedDict):
    version: int
    model_type: str
    prediction_type: str | None
    effective_prediction_type: NotRequired[str | None]
    baseline: float | None
    baseline_mode: str
    training_params: ModelParams
    retained_iterations: NotRequired[int | None]


class ClassificationState(TypedDict):
    version: int
    model_type: str
    prediction_type: str
    baseline: float
    baseline_mode: str
    threshold: float | None
    validation_f1: float | None
    pu_loss: str | None
    class_prior: float | None
    best_iteration: int | None
    training_params: ModelParams
    retained_iterations: NotRequired[int | None]


class InflationState(TypedDict):
    version: int
    model_type: str
    baseline_mode: str
    regressor: RegressionState
    inflation: InflationConfig
    reference_date: str
    prediction_date_column: str


ModelState = RegressionState | ClassificationState | InflationState


def read_state(
    context: Any,
    expected_type: str,
    baseline_mode: str,
    versions: tuple[int, ...] = (1,),
) -> dict[str, object]:
    state = cast(
        dict[str, object], json.loads(Path(context.artifacts["config"]).read_text())
    )
    if state.get("version") not in versions or state.get("model_type") != expected_type:
        raise ValueError("Unsupported model state version or model type")
    if state.get("baseline_mode") != baseline_mode:
        raise ValueError("Invalid baseline application mode")
    return state


def upgrade_classification_state(state: dict[str, object]) -> ClassificationState:
    """Return version-2 classification state from a version-1 or -2 payload.

    Version 1 (6.x and early 7.0 development artifacts) stored the PU prior as
    ``pu_prior``; version 2 calls it ``class_prior``. ``training_params`` stay
    untouched, so refitting an old PU model fails loudly on ``pu_prior``.
    """
    upgraded = dict(state)
    if upgraded["version"] == 1:
        upgraded["class_prior"] = upgraded.pop("pu_prior")
        upgraded["version"] = 2
    return cast(ClassificationState, upgraded)
