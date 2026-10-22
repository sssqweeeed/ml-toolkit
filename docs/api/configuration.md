# Configuration

`TrainingConfig` controls tuning, shared reporting and model packaging.
A [splitter](training.md#chronological-splits) supplies validation and final-fit data. Model factories, metrics, selectors and
report objects are explicit `train_model` arguments.

## Start with a small configuration

<!-- example: configuration-small -->

```python
from dataclasses import replace
from ml_toolkit.config import ReportingConfig, TrackingConfig, TrainingConfig
from ml_toolkit.evaluation import ObservedFitted, TimeObservedFitted

config = TrainingConfig(tracking=TrackingConfig(log_final_model=False))
with_reports = replace(config, reporting=ReportingConfig(psi=True))
prediction_evaluators = [ObservedFitted(), TimeObservedFitted(period="month")]
```

Pass `prediction_evaluators` separately when training. Add or remove objects to
choose reports. PSI and SHAP are off by default. `on_tuning=False`,
`on_validation=True`, and `on_final=True` gate all additional reporting, including
custom metrics; the primary metric always runs. Every validation fold follows
the same flags. Evaluators choose which split labels to use themselves.

Configurations are immutable keyword-only dataclasses; use `replace` to change
them. Shared `features` and `n_bins` define train-fitted levels for PSI and
observed/fitted. `features=()` disables these
level reports. `period` applies to PSI; temporal evaluators have their own period.

See the [complete task recipes](../configuration.md) and
[custom evaluators](training.md#add-custom-evaluation).

## API reference

::: ml_toolkit.config.TrainingConfig
::: ml_toolkit.config.TuningConfig
::: ml_toolkit.config.ReportingConfig
::: ml_toolkit.config.FeatureLevelConfig
::: ml_toolkit.config.ShapConfig
::: ml_toolkit.config.FigureConfig
::: ml_toolkit.config.TrackingConfig

`TimePeriod` supports minute, hour, day, week, month, quarter and year. Temporal
reports preserve input timezones. Standalone PSI also accepts these settings
through its [own API](psi.md).
