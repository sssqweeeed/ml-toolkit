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


class ClassificationState(TypedDict):
    version: int
    model_type: str
    prediction_type: str
    baseline: float
    baseline_mode: str
    threshold: float | None
    validation_f1: float | None
    pu_loss: str | None
    pu_prior: float | None
    best_iteration: int | None
    training_params: ModelParams


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
    context: Any, expected_type: str, baseline_mode: str
) -> dict[str, object]:
    state = cast(
        dict[str, object], json.loads(Path(context.artifacts["config"]).read_text())
    )
    if state.get("version") != 1 or state.get("model_type") != expected_type:
        raise ValueError("Unsupported model state version or model type")
    if state.get("baseline_mode") != baseline_mode:
        raise ValueError("Invalid baseline application mode")
    return state
