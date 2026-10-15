# Configuration

Configuration objects describe the workflow: how to split time, which optional
stages to run, what to report, and which models to package. Data, model
factories, metric functions, and search-space callbacks are passed separately
to `train_model`.

## Start with a small configuration

```python
from dataclasses import replace

from ml_toolkit.config import (
    HoldoutConfig,
    ReportingConfig,
    RegressionDiagnosticsConfig,
    TimeSeriesCVConfig,
    TrackingConfig,
    TrainingConfig,
)

config = TrainingConfig(
    validation=HoldoutConfig(val_size=0.2),
    test_size=0.2,
    tracking=TrackingConfig(log_final_model=False),
)
with_reports = replace(
    config,
    reporting=ReportingConfig(
        diagnostics=RegressionDiagnosticsConfig(task="frequency"),
    ),
)
with_cv = replace(with_reports, validation=TimeSeriesCVConfig(n_splits=3))
```

`config` enables a validation holdout and a future test tail. Feature selection,
tuning, and built-in diagnostic figures are off by default. `with_reports` adds
frequency diagnostics; `with_cv` switches to three expanding validation windows.
The original configurations keep their values.

## Change only what you need

Configurations are immutable, keyword-only dataclasses. Use `replace` for
changes, including nested objects. Set `test_size=None` to use all rows for
training and validation, or `reporting.diagnostics=None` to disable built-in
diagnostics. Model packaging defaults to on for the final fit; disabling it
still allows MLflow metrics and configuration logging.

The [task recipes](../configuration.md) expand every field when you need a
complete configuration to adapt.

## API reference

Import configurations from `ml_toolkit.config`.

::: ml_toolkit.config
