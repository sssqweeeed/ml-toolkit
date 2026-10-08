"""Schemas for persisted native-model serving state."""

from typing import TypedDict

from ..inflation import InflationConfig
from ..types import ModelParams


class RegressionState(TypedDict):
    version: int
    model_type: str
    prediction_type: str | None
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
