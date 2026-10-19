from typing import Any

class PythonModelContext:
    artifacts: dict[str, str]

class PythonModel:
    def load_context(self, context: Any) -> None: ...
    def predict(
        self, context: Any, model_input: Any, params: dict[str, Any] | None = ...
    ) -> object: ...
