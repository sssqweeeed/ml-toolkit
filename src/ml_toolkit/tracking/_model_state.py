"""Auditable state for Models From Code."""

from collections.abc import Mapping

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol, cast

import catboost
import pandas as pd
from mlflow.models.model import ModelInfo

from ..types import FloatArray
from .model_artifacts import log_model_from_code
from .state import ModelState


class NativeModelState(Protocol):
    @property
    def native_model(self) -> catboost.CatBoost: ...
    def export_state(self) -> ModelState: ...


def log_catboost(
    model: NativeModelState,
    sample_frame: pd.DataFrame,
    sample_prediction: FloatArray,
    *,
    name: str,
    registered_model_name: str | None = None,
    conda_env: Path | None = None,
    metadata: dict[str, object] | None = None,
    extra_artifacts: Mapping[str, str] | None = None,
) -> ModelInfo:
    state = model.export_state()
    with TemporaryDirectory(prefix="ml-toolkit-model-") as directory:
        directory = Path(directory)
        model_path = directory / "model.cbm"
        model.native_model.save_model(str(model_path))
        config_path = directory / "config.json"
        serialized_state = json.dumps(state, default=str)
        config_path.write_text(serialized_state, encoding="utf-8")
        logged_state = cast(dict[str, object], json.loads(serialized_state))
        source = (
            Path(__file__).resolve().parent
            / "entrypoints"
            / f"{type(model).__name__}.py"
        )
        return log_model_from_code(
            source=source,
            sample_frame=sample_frame,
            sample_prediction=sample_prediction,
            artifacts={"model": str(model_path), "config": str(config_path)},
            extra_artifacts=extra_artifacts,
            metadata={**logged_state, **(metadata or {})},
            params=model.native_model.get_all_params(),
            name=name,
            registered_model_name=registered_model_name,
            conda_env=conda_env,
        )
