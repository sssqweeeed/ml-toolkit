from typing import Any

from .model import ModelInfo as ModelInfo
from .signature import ModelSignature as ModelSignature

def infer_signature(
    model_input: object = ...,
    model_output: object = ...,
    params: dict[str, object] | None = ...,
) -> ModelSignature: ...
def set_model(model: object) -> None: ...
