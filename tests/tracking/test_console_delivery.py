"""Console fallback follows actual delivery without rerunning logging filters."""

import io
import logging
from pathlib import Path
import sys

import pytest

from ml_toolkit._console import Logger, capture_messages, output_context


@pytest.fixture
def application(monkeypatch, caplog):
    logger = logging.getLogger("ml_toolkit.console_delivery")
    caplog.set_level(logging.DEBUG, logger=logger.name)
    monkeypatch.setattr(logger, "handlers", [])
    monkeypatch.setattr(logger, "filters", [])
    monkeypatch.setattr(logger, "propagate", False)
    yield logger
    for handler in logger.handlers:
        handler.close()


@pytest.mark.parametrize("kind", ["file", "null", "capture"])
def test_non_console_handlers_preserve_fallback(application, capsys, tmp_path, kind):
    stream = io.StringIO()
    handler = {
        "file": lambda: logging.FileHandler(tmp_path / "run.log"),
        "null": logging.NullHandler,
        "capture": lambda: logging.StreamHandler(stream),
    }[kind]()
    application.addHandler(handler)
    with output_context("debug"):
        Logger(application.name).info("same milestone %s", 1)
    assert capsys.readouterr().err == "same milestone 1\n"
    if kind == "file":
        assert (tmp_path / "run.log").read_text() == "same milestone 1\n"
    elif kind == "capture":
        assert stream.getvalue() == "same milestone 1\n"


@pytest.mark.parametrize("filter_at", ["logger", "handler"])
def test_stateful_filters_run_once_and_rejected_records_have_fallback(
    application, capsys, filter_at
):
    calls = []

    def alternate(record):
        calls.append(record.getMessage())
        return len(calls) % 2 == 1

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("APP %(message)s"))
    application.addHandler(handler)
    (application if filter_at == "logger" else handler).addFilter(alternate)
    with output_context("debug"):
        logger = Logger(application.name)
        logger.info("first")
        logger.info("second")
    assert calls == ["first", "second"]
    assert capsys.readouterr().err == "APP first\nsecond\n"


@pytest.mark.skipif(sys.version_info < (3, 12), reason="Replacement filters require Python 3.12")
@pytest.mark.parametrize("filter_at", ["logger", "handler"])
def test_replacement_records_are_delivered_once(application, capsys, filter_at):
    calls = []

    def replace_record(record):
        calls.append(record)
        return logging.makeLogRecord({**record.__dict__, "msg": "filtered", "args": ()})

    handler = logging.StreamHandler(sys.stderr)
    application.addHandler(handler)
    (application if filter_at == "logger" else handler).addFilter(replace_record)
    with output_context("normal"), capture_messages() as delivered:
        Logger(application.name).info("original")
    assert len(calls) == 1
    assert capsys.readouterr().err == "filtered\n"
    assert delivered == {"original"}


def test_origin_levels_and_propagation_boundaries(application, capsys, monkeypatch):
    parent = logging.getLogger("ml_toolkit")
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("APP %(message)s"))
    monkeypatch.setattr(parent, "handlers", [handler])
    monkeypatch.setattr(parent, "propagate", False)
    # Ancestor logger filters and levels do not filter propagated records.
    monkeypatch.setattr(parent, "level", logging.CRITICAL)
    monkeypatch.setattr(parent, "filters", [lambda record: False])
    logger = Logger(application.name)
    with output_context("debug"):
        logger.info("stopped")
        monkeypatch.setattr(application, "propagate", True)
        logger.info("propagated")
        monkeypatch.setattr(application, "disabled", True)
        logger.info("disabled")
    assert capsys.readouterr().err == "stopped\nAPP propagated\ndisabled\n"


def test_record_factory_and_caller_are_preserved(application, monkeypatch):
    records = []

    class Recorder(logging.Handler):
        def emit(self, record):
            records.append(record)

    factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = factory(*args, **kwargs)
        record.request_id = "request-1"
        return record

    monkeypatch.setattr(logging, "_logRecordFactory", record_factory)
    application.addHandler(Recorder())
    with output_context("normal"):
        expected_line = sys._getframe().f_lineno + 1
        Logger(application.name).info("milestone %s", 1)
    assert len(records) == 1
    record = records[0]
    assert record.request_id == "request-1"
    assert Path(record.pathname) == Path(__file__)
    assert record.lineno == expected_line
    assert record.funcName == "test_record_factory_and_caller_are_preserved"
    assert record.msg == "milestone %s" and record.args == (1,)


def test_last_resort_and_repeated_events(application, capsys, monkeypatch):
    handler = logging.StreamHandler(sys.stderr)
    monkeypatch.setattr(logging, "lastResort", handler)
    with output_context("normal"):
        logger = Logger(application.name)
        logger.info("repeated")
        logger.info("repeated")
    assert capsys.readouterr().err == "repeated\nrepeated\n"


def test_handler_can_stop_propagation_for_current_record(application, capsys, monkeypatch):
    parent = logging.getLogger("ml_toolkit")
    parent_handler = logging.StreamHandler(sys.stderr)
    parent_handler.setFormatter(logging.Formatter("PARENT %(message)s"))
    monkeypatch.setattr(parent, "handlers", [parent_handler])
    monkeypatch.setattr(application, "propagate", True)

    class StopPropagation(logging.Handler):
        def emit(self, record):
            application.propagate = False

    application.addHandler(StopPropagation())
    with output_context("normal"):
        Logger(application.name).info("milestone")
    assert capsys.readouterr().err == "milestone\n"


def test_missing_handler_diagnostic_remains_once(application, capsys, monkeypatch):
    monkeypatch.setattr(application, "handlers", [])
    monkeypatch.setattr(logging, "lastResort", None)
    monkeypatch.setattr(logging, "raiseExceptions", True)
    monkeypatch.setattr(application.manager, "emittedNoHandlerWarning", False)
    with output_context("normal"):
        Logger(application.name).info("first")
        Logger(application.name).info("second")
    output = capsys.readouterr().err
    assert output.count("No handlers could be found") == 1
    assert output.endswith("first\nsecond\n")


def test_capture_restores_after_failure(application, capsys):
    application.addHandler(logging.StreamHandler(sys.stderr))
    with output_context("normal"), capture_messages() as outer:
        with pytest.raises(RuntimeError), capture_messages():
            raise RuntimeError("failed selection")
        Logger(application.name).info("outer")
    assert outer == {"outer"}
    assert capsys.readouterr().err == "outer\n"
