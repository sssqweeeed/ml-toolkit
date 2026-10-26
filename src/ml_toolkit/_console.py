"""Run-local console output, independent of application logging configuration."""

from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
import logging
import sys
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .data import Pool

Verbosity = Literal["silent", "normal", "verbose", "debug"]
_LEVELS: tuple[Verbosity, ...] = ("silent", "normal", "verbose", "debug")
_active: ContextVar[Verbosity | None] = ContextVar("ml_toolkit_verbosity", default=None)


@contextmanager
def output_context(verbosity: Verbosity) -> Generator[None, None, None]:
    token = _active.set(verbosity)
    try:
        yield
    finally:
        _active.reset(token)


def enabled(verbosity: Verbosity) -> bool:
    active = _active.get()
    return (
        active is not None
        and active != "silent"
        and _LEVELS.index(active) >= _LEVELS.index(verbosity)
    )


def message(text: str, *args: object, verbosity: Verbosity = "normal") -> None:
    if enabled(verbosity):
        print(text % args if args else text, file=sys.stderr, flush=True)


class Logger:
    """Forward records unchanged and optionally mirror them to this run's console."""

    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)

    @property
    def name(self) -> str:
        return self._logger.name

    def info(self, text: str, *args: object, console: bool = True) -> None:
        self._logger.info(text, *args, stacklevel=2)
        if console:
            message(text, *args)

    def debug(
        self,
        text: str,
        *args: object,
        console: bool = True,
        verbosity: Verbosity = "debug",
    ) -> None:
        self._logger.debug(text, *args, stacklevel=2)
        if console:
            message(text, *args, verbosity=verbosity)


def _cell(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return format(value, ".6g")
    return str(value).replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def _table(
    title: str, headers: Sequence[str], rows: Sequence[Sequence[object]]
) -> None:
    cells = [[_cell(value) for value in row] for row in (headers, *rows)]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]
    lines = [title]
    for i, row in enumerate(cells):
        lines.append(
            " | ".join(
                value.ljust(width) for value, width in zip(row, widths, strict=True)
            ).rstrip()
        )
        if i == 0:
            lines.append("-+-".join("-" * width for width in widths))
    print("\n".join(lines), file=sys.stderr, flush=True)


def datasets(title: str, samples: Mapping[str, "Pool"]) -> None:
    if not enabled("verbose"):
        return
    _table(
        title,
        ("sample", "rows", "features", "from", "to"),
        [
            (name, len(pool), len(pool.X.columns), pool.time[0], pool.time[-1])
            for name, pool in samples.items()
        ],
    )


def metrics(
    title: str,
    samples: Mapping[str, Mapping[str, float]],
    *,
    verbosity: Verbosity = "verbose",
) -> None:
    if not enabled(verbosity):
        return
    names = dict.fromkeys(name for scores in samples.values() for name in scores)
    if not names:
        message("%s: no metrics", title, verbosity=verbosity)
        return
    _table(
        title,
        ("metric", *samples),
        [(name, *(scores.get(name) for scores in samples.values())) for name in names],
    )
