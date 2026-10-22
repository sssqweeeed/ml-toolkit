# Configuration

`TrainingConfig` controls tuning, shared reporting, model packaging and optional
training including test via `fit_on_test=False`. Enabling this flag returns an
additional `result.final_model` and requires a held-out test sample. See the
[training scenarios](training.md#optionally-train-including-test).
A [splitter](splitting.md) supplies validation and final-fit data. Model factories, metrics, selectors and
report objects are explicit `train_model` arguments.

## Start with a small configuration

<!-- example: configuration-small -->

```python
from dataclasses import replace
from ml_toolkit.config import ReportingConfig, TrackingConfig, TrainingConfig
from ml_toolkit.evaluation import ObservedFitted, TimeObservedFitted

config = TrainingConfig(tracking=TrackingConfig(log_test_model=False, log_final_model=False))
with_reports = replace(config, reporting=ReportingConfig(psi=True))
prediction_evaluators = [ObservedFitted(), TimeObservedFitted(period="month")]
```

Configurations are immutable keyword-only dataclasses; `replace` produces a
modified copy. Pass evaluator objects separately to select individual reports.
Their execution, sample selection and shared feature levels are described in
[Evaluation and plots](../evaluation/index.md).

| Setting | Responsibility |
| --- | --- |
| `TrainingConfig` | Tuning, reporting and model packaging settings |
| `TuningConfig` | Objective name/direction, trial budget and seed |
| `ReportingConfig` | Stage switches, PSI/SHAP and common report options |
| `FeatureLevelConfig` | Numerical or categorical feature levels and bounds |
| `FigureConfig` | Figure size, resolution and output format |
| `ShapConfig` | Explanation sampling and displayed features |
| `TrackingConfig` | Which models to package and how to package them |

See [Task recipes](../configuration.md) for complete configuration in a training
call and [Build an evaluator](../evaluation/custom.md) to add your own report.

## API reference

::: ml_toolkit.config.TrainingConfig
    options:
      members: []
::: ml_toolkit.config.TuningConfig
    options:
      members: []
::: ml_toolkit.config.ReportingConfig
    options:
      members: []
::: ml_toolkit.config.FeatureLevelConfig
    options:
      members: []
::: ml_toolkit.config.ShapConfig
    options:
      members: []
::: ml_toolkit.config.FigureConfig
    options:
      members: []
::: ml_toolkit.config.TrackingConfig
    options:
      members: []

`TimePeriod` supports minute, hour, day, week, month, quarter and year. Temporal
reports preserve input timezones. Standalone PSI also accepts these settings
through its [own API](psi.md).
