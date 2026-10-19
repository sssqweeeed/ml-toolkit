from typing import Any

from ..models.model import ModelInfo
from .model import PythonModel as PythonModel
from .model import PythonModelContext as PythonModelContext

class PyFuncModel:
    def predict(self, data: Any, params: dict[str, object] | None = ...) -> Any: ...

def load_model(model_uri: str) -> PyFuncModel: ...

def log_model(
    *,
    python_model: str,
    name: str,
    registered_model_name: str | None,
    signature: object,
    input_example: object,
    params: dict[str, object] | None = ...,
    artifacts: dict[str, str],
    metadata: dict[str, object] | None = ...,
    code_paths: list[str] | None = ...,
    conda_env: str | None = ...,
    pip_requirements: list[str] | None = ...,
) -> ModelInfo: ...
