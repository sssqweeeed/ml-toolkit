# Configuration

`TrainingConfig` controls tuning, shared reporting and model packaging.
A [splitter](splitting.md) supplies validation and final-fit data. Model factories, metrics, selectors and
report objects are explicit `train_model` arguments.

## Start with a small configuration

<!-- example: configuration-small -->

```python
from dataclasses import replace
from ml_toolkit.config import ReportingConfig, TrackingConfig, TrainingConfig
from ml_toolkit.evaluation import ObservedFitted, TimeObservedFitted
from ml_toolkit.data_analysis import NumericalFeatureDistributions

config = TrainingConfig(tracking=TrackingConfig(log_final_model=False))
with_reports = replace(config, reporting=ReportingConfig(psi=True))
prediction_evaluators = [ObservedFitted(), TimeObservedFitted(period="month")]
data_analyzers = [NumericalFeatureDistributions(period="month")]
```

Configurations are immutable keyword-only dataclasses; `replace` produces a
modified copy. Pass evaluator objects separately to select individual reports.
Their execution, sample selection and shared feature levels are described in
[Evaluation](../evaluation/index.md) and [Data analysis](../data-analysis/index.md).
Pass `prediction_evaluators=` and `data_analyzers=` separately to `train_model`.
PSI and O/F share `ReportingConfig.features` and `n_bins`; each independently
fits a `LevelTransformer` on the same training rows. See
[how levels are shared](psi.md#how-psi-and-observedfitted-share-levels).

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
call, [Build an evaluator](../evaluation/custom.md) for custom prediction reports,
and [Build a data analyzer](../data-analysis/custom.md) for custom data reports.

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
