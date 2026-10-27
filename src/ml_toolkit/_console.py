"""Run-local console output, independent of application logging configuration."""

from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
import logging
import sys
from typing import TYPE_CHECKING, Literal, TextIO, cast

if TYPE_CHECKING:
    from .data import Pool

Verbosity = Literal["silent", "normal", "verbose", "debug"]
_LEVELS: tuple[Verbosity, ...] = ("silent", "normal", "verbose", "debug")
_active: ContextVar[Verbosity | None] = ContextVar("ml_toolkit_verbosity", default=None)
_captured: ContextVar[set[str] | None] = ContextVar("ml_toolkit_console_messages", default=None)


@contextmanager
def output_context(verbosity: Verbosity) -> Generator[None, None, None]:
    token = _active.set(verbosity)
    try:
        yield
    finally:
        _active.reset(token)


@contextmanager
def capture_messages() -> Generator[set[str], None, None]:
    messages: set[str] = set()
    token = _captured.set(messages)
    try:
        yield messages
    finally:
        _captured.reset(token)


def enabled(verbosity: Verbosity) -> bool:
    active = _active.get()
    return (
        active is not None
        and active != "silent"
        and _LEVELS.index(active) >= _LEVELS.index(verbosity)
    )


def message(text: str, *args: object, verbosity: Verbosity = "normal") -> None:
    if enabled(verbosity):
        rendered = text % args if args else text
        print(rendered, file=sys.stderr, flush=True)
        captured = _captured.get()
        if captured is not None:
            captured.add(rendered)


def _handlers(source: logging.Logger) -> Generator[logging.Handler, None, None]:
    current: logging.Logger | None = source
    found = False
    while current is not None:
        for handler in current.handlers:
            found = True
            yield handler
        current = current.parent if current.propagate else None
    if not found:
        if logging.lastResort is not None:
            yield logging.lastResort
        elif logging.raiseExceptions and not source.manager.emittedNoHandlerWarning:
            sys.stderr.write(f'No handlers could be found for logger "{source.name}"\n')
            source.manager.emittedNoHandlerWarning = True


class Logger:
    """Forward records and print run output when console handlers did not deliver it."""

    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)

    @property
    def name(self) -> str:
        return self._logger.name

    def info(self, text: str, *args: object, console: bool = True) -> None:
        delivered = self._log(logging.INFO, text, args)
        if console and not delivered:
            message(text, *args)

    def debug(
        self,
        text: str,
        *args: object,
        console: bool = True,
        verbosity: Verbosity = "debug",
    ) -> None:
        delivered = self._log(logging.DEBUG, text, args)
        if console and not delivered:
            message(text, *args, verbosity=verbosity)

    def _log(self, level: int, text: str, args: tuple[object, ...]) -> bool:
        if _active.get() in (None, "silent"):
            self._logger.log(level, text, *args, stacklevel=3)
            return False
        if not self._logger.isEnabledFor(level):
            return False
        filename, line, function, stack = self._logger.findCaller(stacklevel=3)
        record = self._logger.makeRecord(
            self.name, level, filename, line, text, args, None, function, sinfo=stack
        )
        # Dispatch the actual record once: probing filters before logging would
        # run stateful filters twice and miss replacement records on Python 3.12.
        filtered = self._logger.filter(record)
        if not filtered:
            return False
        if isinstance(filtered, logging.LogRecord):
            record = filtered
        delivered = False
        for handler in _handlers(self._logger):
            if record.levelno < handler.level:
                continue
            accepted = handler.handle(record)
            if (
                accepted
                and isinstance(handler, logging.StreamHandler)
                and not isinstance(handler, logging.FileHandler)
                and any(
                    cast("logging.StreamHandler[TextIO]", handler).stream is stream
                    for stream in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__)
                )
            ):
                delivered = True
        captured = _captured.get()
        if delivered and captured is not None:
            captured.add(text % args if args else text)
        return delivered


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
