"""Small MLflow operations used by the training workflow."""

import json
from pathlib import Path

import mlflow
from mlflow.tracking.fluent import ActiveRun

from ..types import ModelParams


def run(name: str) -> ActiveRun:
    return mlflow.start_run(run_name=name, nested=mlflow.active_run() is not None)


def log_params(params: ModelParams) -> None:
    mlflow.log_params(params)


def log_metrics(metrics: dict[str, float]) -> None:
    mlflow.log_metrics(metrics)


def log_json(payload: object, artifact_file: str) -> None:
    mlflow.log_text(json.dumps(payload, default=str, indent=2), artifact_file)


def log_artifact(path: Path, *, artifact_path: str) -> None:
    mlflow.log_artifact(str(path), artifact_path=artifact_path)
