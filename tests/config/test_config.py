"""Immutable configuration contracts, ranges and public field documentation."""

import inspect
import re
from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
from datetime import date, datetime

import pytest

import ml_toolkit.config as settings
from ml_toolkit.config import (
    ClassificationDiagnosticsConfig,
    CommonPlotsConfig,
    FigureConfig,
    HoldoutConfig,
    NumericalFeatureDistributionConfig,
    PredictionEvaluationConfig,
    PUDiagnosticsConfig,
    PUPlotsConfig,
    ReportingConfig,
    ShapConfig,
    TimeSeriesCVConfig,
    TrainingConfig,
    TuningConfig,
)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TimeSeriesCVConfig(n_splits=1),
        lambda: TimeSeriesCVConfig(n_splits=3, skip_splits=3),
        lambda: TimeSeriesCVConfig(n_splits=3, gap=-1),
        lambda: TuningConfig(metric="", direction="minimize"),
        lambda: TuningConfig(metric="mae", direction="minimize", timeout_seconds=0),
        lambda: CommonPlotsConfig(epsilon=0),
        lambda: CommonPlotsConfig(n_bins=0),
        lambda: FigureConfig(figsize=(0, 5)),
        lambda: ShapConfig(dependence_features=("x", "x")),
        lambda: ReportingConfig(validation_splits=()),
        lambda: ReportingConfig(validation_splits=(0,)),
        lambda: NumericalFeatureDistributionConfig(quantile_bands=((0.9, 0.1),)),
        lambda: PredictionEvaluationConfig(stages=("validation", "validation")),
        lambda: TrainingConfig(
            validation=TimeSeriesCVConfig(n_splits=3, skip_splits=1),
            reporting=ReportingConfig(
                diagnostics=PUDiagnosticsConfig(), validation_splits=(1,)
            ),
        ),
    ],
    ids=[
        "too-few-splits", "skip-all-splits", "negative-gap", "empty-objective",
        "zero-timeout", "zero-smoothing", "zero-bins", "zero-figure-width",
        "duplicate-shap-features", "no-report-splits", "zero-report-split",
        "reversed-quantiles", "duplicate-prediction-evaluation-stages", "skipped-report-split",
    ],
)
def test_invalid_ranges_fail_on_construction(factory):
    with pytest.raises(ValueError):
        factory()


def test_configs_are_immutable_keyword_only_and_replaceable():
    original = TrainingConfig(validation=TimeSeriesCVConfig(n_splits=3))
    with pytest.raises(FrozenInstanceError):
        original.validation.gap = 7
    changed = replace(original, validation=replace(original.validation, gap=7))
    assert original.validation.gap == 0 and changed.validation.gap == 7
    with pytest.raises(TypeError, match="positional"):
        TimeSeriesCVConfig(3)
    assert "gains" not in {f.name for f in fields(PUPlotsConfig)}
    assert PUDiagnosticsConfig().task == "pu_classification"
    assert ClassificationDiagnosticsConfig().task == "classification"


@pytest.mark.parametrize("cutoff", ["2024-02-29", date(2024, 2, 29)])
def test_test_start_date_is_normalized_and_replaceable(cutoff):
    config = TrainingConfig(validation=HoldoutConfig(), test_size=cutoff)
    assert config.test_size == date(2024, 2, 29)
    assert replace(config, test_size="2025-01-01").test_size == date(2025, 1, 1)
    assert config.test_size == date(2024, 2, 29)
    with pytest.raises(FrozenInstanceError):
        config.test_size = date(2025, 1, 1)


@pytest.mark.parametrize(
    "cutoff",
    [
        "",
        "invalid",
        "2025-02-29",
        "2025-13-01",
        "2025-01-00",
        "20250101",
        "2025-W01-1",
        "2025-1-1",
        " 2025-01-01",
        "2025-01-01 ",
        "2025-01-01T12:00:00",
    ],
)
def test_invalid_test_start_date_fails_at_configuration(cutoff):
    with pytest.raises(ValueError, match="test_size date must use YYYY-MM-DD"):
        TrainingConfig(validation=HoldoutConfig(), test_size=cutoff)


@pytest.mark.parametrize(
    "diagnostics", [None, ClassificationDiagnosticsConfig(plots=None)]
)
@pytest.mark.parametrize(
    "validation, selected",
    [
        (HoldoutConfig(), (2,)),
        (TimeSeriesCVConfig(n_splits=3, skip_splits=1), (1,)),
    ],
)
def test_disabled_diagnostics_allow_unavailable_split_indices(
    diagnostics, validation, selected
):
    config = TrainingConfig(
        validation=validation,
        reporting=ReportingConfig(
            diagnostics=diagnostics, validation_splits=selected
        ),
    )
    assert config.reporting.validation_splits == selected
    with pytest.raises(ValueError, match="evaluated validation split indices"):
        replace(
            config,
            reporting=replace(
                config.reporting, diagnostics=ClassificationDiagnosticsConfig()
            ),
        )


def test_prediction_evaluation_policy_is_optional_without_changing_explicit_settings():
    assert ReportingConfig().prediction_evaluation is None
    assert ReportingConfig(prediction_evaluation=None).prediction_evaluation is None
    policy = PredictionEvaluationConfig(stages=())
    assert ReportingConfig(prediction_evaluation=policy).prediction_evaluation is policy


def test_exact_test_start_time_is_rejected():
    with pytest.raises(ValueError, match="calendar date without a time"):
        TrainingConfig(validation=HoldoutConfig(), test_size=datetime(2025, 1, 1))


def test_config_docstrings_describe_every_public_field():
    for name in settings.__all__:
        cls = getattr(settings, name)
        if not inspect.isclass(cls) or not is_dataclass(cls):
            continue
        doc = inspect.getdoc(cls)
        assert "Parameters\n----------" in doc, name
        for item in fields(cls):
            assert re.search(rf"^{item.name} :", doc, re.MULTILINE), (name, item.name)
