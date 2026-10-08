from types import TracebackType
from typing import Self

class RunInfo:
    run_id: str

class ActiveRun:
    info: RunInfo
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...
