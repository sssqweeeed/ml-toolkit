"""Auditable state for Models From Code."""

from dataclasses import asdict, is_dataclass
from datetime import date, datetime
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .model_artifacts import log_model_from_code


def audit_value(value):
    if is_dataclass(value) and not isinstance(value, type):
        return audit_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): audit_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [audit_value(item) for item in value]
    if isinstance(value, np.generic):
        return audit_value(value.item())
    if isinstance(value, (Path, date, datetime)):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return {"python_type": f"{type(value).__module__}.{type(value).__qualname__}"}


def log_catboost(
    model,
    sample_frame,
    sample_prediction,
    *,
    name,
    registered_model_name=None,
    conda_env=None,
    metadata=None,
):
    state = model._export_state()
    with TemporaryDirectory(prefix="ml-toolkit-model-") as directory:
        directory = Path(directory)
        model_path = directory / "model.cbm"
        model.native_model.save_model(str(model_path))
        config_path = directory / "config.json"
        config_path.write_text(json.dumps(audit_value(state)), encoding="utf-8")
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
            metadata={**state, **(metadata or {})},
            params=audit_value(model.native_model.get_all_params()),
            name=name,
            registered_model_name=registered_model_name,
            conda_env=conda_env,
        )
