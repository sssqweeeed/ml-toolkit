"""Immutable configuration contracts, ranges and public field documentation."""

import inspect
import re
from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
import pytest
import ml_toolkit.config as settings
from ml_toolkit.config import (
    FigureConfig,
    ReportingConfig,
    ShapConfig,
    TrainingConfig,
    TuningConfig,
)


def test_configs_are_immutable_keyword_only_and_replaceable():
    original = TrainingConfig(reporting=ReportingConfig(on_tuning=False))
    with pytest.raises(FrozenInstanceError):
        original.reporting.on_tuning = True
    changed = replace(original, reporting=replace(original.reporting, on_tuning=True))
    assert not original.reporting.on_tuning and changed.reporting.on_tuning
    with pytest.raises(TypeError, match="positional"):
        TrainingConfig(None)
    assert {item.name for item in fields(TrainingConfig)} == {
        "tuning",
        "reporting",
        "tracking",
        "fit_on_test",
        "verbosity",
    }


def test_config_docstrings_describe_every_public_field():
    for name in settings.__all__:
        cls = getattr(settings, name)
        if not inspect.isclass(cls) or not is_dataclass(cls):
            continue
        doc = inspect.getdoc(cls)
        assert "Parameters\n----------" in doc, name
        for item in fields(cls):
            assert re.search(rf"^{item.name} :", doc, re.MULTILINE), (name, item.name)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TuningConfig(metric="", direction="minimize"),
        lambda: TuningConfig(metric="mae", direction="minimize", timeout_seconds=0),
        lambda: ReportingConfig(epsilon=0),
        lambda: ReportingConfig(n_bins=0),
        lambda: FigureConfig(figsize=(0, 5)),
        lambda: ShapConfig(dependence_features=("x", "x")),
    ],
)
def test_invalid_ranges_fail_on_construction(factory):
    with pytest.raises(ValueError):
        factory()
