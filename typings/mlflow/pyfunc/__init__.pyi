from typing import Any

from ..models.model import ModelInfo
from .model import PythonModel as PythonModel

def log_model(
    *,
    python_model: str,
    name: str,
    registered_model_name: str | None,
    signature: object,
    input_example: object,
    params: dict[str, object],
    artifacts: dict[str, str],
    metadata: dict[str, object] | None,
    code_paths: list[str],
    conda_env: str | None = ...,
    pip_requirements: list[str] | None = ...,
) -> ModelInfo: ...
