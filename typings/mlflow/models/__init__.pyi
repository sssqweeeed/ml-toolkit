from typing import Any

from .model import ModelInfo as ModelInfo
from .signature import ModelSignature as ModelSignature

class Model:
    flavors: dict[str, Any]
    @classmethod
    def load(cls, path: str) -> Model: ...

def infer_signature(
    model_input: object = ...,
    model_output: object = ...,
    params: dict[str, object] | None = ...,
) -> ModelSignature: ...
def set_model(model: object) -> None: ...
