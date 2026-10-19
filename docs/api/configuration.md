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

Use the [task recipes](../configuration.md) to choose metrics and diagnostic
settings, or the [training examples](training.md#select-features-and-tune-parameters)
to wire feature selection and tuning into a run.

## Find a setting

| Change | Configuration |
| --- | --- |
| Validation windows or gaps | `HoldoutConfig`, `TimeSeriesCVConfig` |
| Test tail, feature selection, tuning | `TrainingConfig`, `RFEFeatureSelectionConfig`, `TuningConfig` |
| Enabled reports and evaluated windows | `ReportingConfig` |
| Classification, regression, or PU figures | Corresponding diagnostics and plots configs |
| Shared PSI, observed/fitted, and numerical distributions | `CommonPlotsConfig`, `NumericalFeatureDistributionConfig` |
| SHAP sample size and displayed features | `ShapConfig` |
| Image size, DPI, and format | `FigureConfig` |
| Custom evaluator stages and splits | `PostFitEvaluationConfig` |
| Model packaging and registry | `TrackingConfig` |

Shared plot defaults apply when task diagnostics are enabled; a default
`TrainingConfig` has no built-in diagnostics. `CommonPlotsConfig.features=()`
disables feature-level PSI and observed/fitted, while numerical distributions
and SHAP have their own switches. Set the enclosing diagnostics' `plots=None`
to disable every figure and PSI table.

## API reference

Import configurations from `ml_toolkit.config`.

::: ml_toolkit.config
    options:
      members:
        - HoldoutConfig
        - TimeSeriesCVConfig
        - TrainingConfig
        - RFEFeatureSelectionConfig
        - TuningConfig
        - ReportingConfig
        - ClassificationDiagnosticsConfig
        - RegressionDiagnosticsConfig
        - PUDiagnosticsConfig
        - ClassificationPlotsConfig
        - RegressionPlotsConfig
        - PUPlotsConfig
        - CommonPlotsConfig
        - NumericalFeatureDistributionConfig
        - ShapConfig
        - FigureConfig
        - PostFitEvaluationConfig
        - TrackingConfig
        - FeatureLevelSpec

`DiagnosticsConfig` is the union of the three task diagnostics configurations
above. See the [feature-level reference](psi.md) for fitting and reusing learned
level definitions.
