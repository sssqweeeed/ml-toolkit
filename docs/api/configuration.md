# Configuration

`TrainingConfig` controls console verbosity, tuning, shared reporting, model packaging and optional
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
from ml_toolkit.data_analysis import NumericalFeatureDistributions

config = TrainingConfig(tracking=TrackingConfig(log_test_model=False, log_final_model=False))
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
| `TrainingConfig` | Console verbosity, tuning, reporting and model packaging settings |
| `TuningConfig` | Objective name/direction, trial budget and seed |
| `ReportingConfig` | Stage switches, PSI/SHAP and common report options |
| `FeatureLevelConfig` | Numerical or categorical feature levels and bounds |
| `FigureConfig` | Figure size, resolution and output format |
| `ShapConfig` | Explanation sampling and displayed features |
| `TrackingConfig` | Which models to package and how to package them |

See [Task recipes](../configuration.md) for complete configuration in a training
call, [Build an evaluator](../evaluation/custom.md) for custom prediction reports,
and [Build a data analyzer](../data-analysis/custom.md) for custom data reports.

## Console verbosity

Set `TrainingConfig(verbosity="normal")` to see training progress without
configuring a Python logger. `normal` is the default. Each setting applies only
to the current `train_model` call, including its selection, tuning and reports.

| Level | Output |
| --- | --- |
| `silent` | No automatic progress or result output from ml-toolkit. Warnings and exceptions remain visible. |
| `normal` | Run name, original dataset row/feature counts, stage and split milestones, selection/tuning summaries, and a final metric table. |
| `verbose` | Everything in `normal`, plus each split's row/feature counts and date range, selected feature count, best iteration, train/validation/test metrics, and each tuning trial's objective. |
| `debug` | Everything in `verbose`, plus selected feature names, built-in selection candidates, and internal model, PSI, SHAP, report and artifact diagnostics. |

<!-- example: configuration-verbosity -->

```python
normal_config = replace(config, verbosity="normal")
detailed_config = replace(config, verbosity="verbose")
quiet_config = replace(config, verbosity="silent")
debug_config = replace(config, verbosity="debug")
```

Pass the chosen configuration as `train_model(..., config=detailed_config)`.
Unknown levels raise `ValueError`. There is no numeric interval or per-tree
progress bar. The selected level is recorded in MLflow's
`configuration/requested.json` together with the other training settings.

The plain-text output goes to **stderr**, including in notebooks and redirected
terminal output. Tables use six significant digits; returned metrics retain
their original precision. A compact final summary looks like this (illustrative
values):

```text
Evaluation summary (test scores belong to the main model)
metric | validation mean | test
-------+-----------------+-----
mae    | 9               | 12
Completed training run example
```

The `test` column is omitted when the splitter has no test sample. With
`fit_on_test=True`, test scores still describe the main model evaluated before
the additional fit. Its train-only metrics appear separately in `verbose` and
`debug`. Missing best iterations are shown as `—`; available iterations use the
model's zero-based `best_iteration`. Undefined existing metrics display as `nan`
or `inf`. Empty metric dictionaries produce a short `no metrics` message.

Dataset counts describe `splitter.data`; per-stage tables describe the actual
pools supplied by the splitter. Splits appear as they are processed, with their
original indices. Output uses existing results and inexpensive metadata; it
does not calculate extra metrics, repeat predictions, or collect all split
pools in memory. Evaluator/analyzer metrics remain in their existing reports.

### Python logging and backend output

Console verbosity does not change CatBoost parameters (such as `verbose=False`),
Optuna verbosity, MLflow output, or experiment recording. Consequently, `silent`
does not suppress messages from those libraries or from your callbacks.

Standard records under the `ml_toolkit` Python logger remain independent and
retain application-controlled levels and handlers. If your application already
prints these records to the terminal, use `verbosity="silent"` to avoid seeing
both the automatic console output and your own handler's messages. Standalone
model, selector, tuning and diagnostic calls keep their existing Python logging
behavior; this setting does not enable automatic output outside `train_model`.

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
