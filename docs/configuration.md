# Configuration by task

Configurations are immutable, keyword-only dataclasses. Start with shared
settings, then choose a model, metric, and diagnostic family for your target.
Use `dataclasses.replace` to change settings without modifying the original.
The [configuration reference](api/configuration.md) lists every field and default.

## Choose a recipe

| Task | Model | Metric | Recipe |
| --- | --- | --- | --- |
| Confirmed binary labels | `CatBoostClassificationModel` | `metrics_classification` | [Binary classification](#binary-classification) |
| General numeric targets, including signed values | `CatBoostRegressionModel` | Custom MAE | [Regression](#regression) |
| Nonnegative frequency targets | `CatBoostLogLinkRegressionModel` | `metrics_frequency` | [Frequency](#frequency) |
| Nonnegative severity targets | `CatBoostLogLinkRegressionModel` | Tweedie metric with fixed power | [Severity](#severity) |
| Known positives mixed with unlabeled observations | `CatBoostPUClassificationModel` | `PUMetric` | [Positive-unlabeled classification](#positiveunlabeled-classification) |

## Imports and shared behavior

Run this block first, then choose one task block below. Each task defines
`config`, `model_factory`, `params`, and `metric` for the same training call.

<!-- example: configuration-shared -->

```python
from dataclasses import replace

from ml_toolkit.config import (
    ClassificationDiagnosticsConfig,
    ClassificationPlotsConfig,
    CommonPlotsConfig,
    FeatureLevelSpec,
    HoldoutConfig,
    PUDiagnosticsConfig,
    PUPlotsConfig,
    RegressionDiagnosticsConfig,
    RegressionPlotsConfig,
    ReportingConfig,
    TimeSeriesCVConfig,
    TrackingConfig,
    TrainingConfig,
)
from ml_toolkit.metrics import (
    MetricScores,
    MetricsSeverity,
    PUMetric,
    metrics_classification,
    metrics_frequency,
    metrics_severity,
)
from ml_toolkit.models import (
    CatBoostClassificationModel,
    CatBoostLogLinkRegressionModel,
    CatBoostPUClassificationModel,
    CatBoostRegressionModel,
)
from ml_toolkit.types import FloatArray

base_config = TrainingConfig(
    validation=TimeSeriesCVConfig(n_splits=3),
    test_size=0.2,
    tracking=TrackingConfig(log_final_model=False),
)
base_params = {
    "iterations": 20,
    "depth": 2,
    "random_seed": 42,
    "thread_count": 2,
    "verbose": False,
    "allow_writing_files": False,
}
common = CommonPlotsConfig()
```

These examples keep feature selection, tuning, SHAP, and model packaging off.
Built-in diagnostics run on the last validation split and on test. Metrics and
configuration are always logged to MLflow. Set
`TrackingConfig(log_final_model=True)` to also package the final model.

- Rows must have ascending timestamps; splits never shuffle. Integers specify
  row counts, and fractions round upward. Test fractions use the original data;
  validation fractions use the retained non-test rows. `gap` and `max_train_size`
  count rows.
- Every validation split independently selects features and tunes parameters
  when enabled. Final fitting uses the last split's features, parameters,
  iteration budget, and classification threshold. It includes former validation
  and gap rows, but excludes test.
- `reporting.validation_splits` selects built-in validation reports only.
  Explicit indices refer to original one-based splits before `skip_splits`.
  Reports do not run during tuning trials.
- Custom evaluation requires aligned `object_id`. Its default stages are
  validation and final, with train/val/test where available. Its scores do not
  enter optimization or validation averages.

## Binary classification

Use a pool whose targets contain confirmed `0` and `1` labels. Include both
classes in evaluated splits for finite ROC AUC; single-class splits return NaN.

<!-- example: configuration-classification -->

```python
classification_config = replace(
    base_config,
    reporting=ReportingConfig(
        diagnostics=ClassificationDiagnosticsConfig(
            plots=ClassificationPlotsConfig(common=common),
        ),
    ),
)
config = classification_config
model_factory = CatBoostClassificationModel
params = {**base_params, "loss_function": "Logloss"}
metric = metrics_classification
```

The reports include precision-recall, F1, calibration, gains/lift, monthly means,
and the shared feature diagnostics.

## Regression

Ordinary regression supports signed targets. Define a three-argument metric
returning named scores; this example uses exposure-weighted absolute error.

<!-- example: configuration-regression -->

```python
def weighted_mae(
    observed: FloatArray, prediction: FloatArray, weight: FloatArray
) -> MetricScores:
    return MetricScores(
        {"mae": float((abs(observed - prediction) * weight).sum() / weight.sum())}
    )


regression_config = replace(
    base_config,
    reporting=ReportingConfig(
        diagnostics=RegressionDiagnosticsConfig(
            plots=RegressionPlotsConfig(common=common),
        ),
    ),
)
config = regression_config
model_factory = CatBoostRegressionModel
params = {**base_params, "loss_function": "RMSE"}
metric = weighted_mae
```

Gains and lift stay off because they require nonnegative targets with a positive
weighted total. Shared feature diagnostics and monthly observed/fitted remain on.

## Frequency

Use nonnegative targets with a positive weighted total in every evaluated split.
The log-link adapter fits a training-only baseline and returns positive rates.

<!-- example: configuration-frequency -->

```python
frequency_config = replace(
    base_config,
    reporting=ReportingConfig(
        diagnostics=RegressionDiagnosticsConfig(
            task="frequency",
            plots=RegressionPlotsConfig(common=common, gains=True, lift=True),
        ),
    ),
)
config = frequency_config
model_factory = CatBoostLogLinkRegressionModel
params = {**base_params, "loss_function": "Poisson"}
metric = metrics_frequency
```

## Severity

Use a Tweedie power consistent with the model loss and metric. Here `power=1.5`
allows zero targets and requires positive predictions and positive weighted
observed totals. Gamma deviance (`power=2`) requires strictly positive targets.

<!-- example: configuration-severity -->

```python
def severity_metric(
    observed: FloatArray, prediction: FloatArray, weight: FloatArray
) -> MetricsSeverity:
    return metrics_severity(observed, prediction, weight, power=1.5)


severity_config = replace(
    base_config,
    reporting=ReportingConfig(
        diagnostics=RegressionDiagnosticsConfig(
            task="severity",
            plots=RegressionPlotsConfig(common=common, gains=True, lift=True),
        ),
    ),
)
config = severity_config
model_factory = CatBoostLogLinkRegressionModel
params = {**base_params, "loss_function": "Tweedie:variance_power=1.5"}
metric = severity_metric
```

## Positive–unlabeled classification

Targets use `1` for known positives and `0` for unlabeled observations. Each
training and evaluated sample needs positive weight in both groups. The prior
is the expected positive fraction in the population, not the labeled fraction.
Use the same prior in the model and metric.

<!-- example: configuration-pu -->

```python
pu_classification_config = replace(
    base_config,
    reporting=ReportingConfig(
        diagnostics=PUDiagnosticsConfig(
            plots=PUPlotsConfig(common=common),
        ),
    ),
)
config = pu_classification_config
model_factory = CatBoostPUClassificationModel
params = {**base_params, "loss": "uPU", "pu_prior": 0.2}
metric = PUMetric(0.2, loss="uPU")
```

PU reports show score distributions and positive capture. They do not treat
unlabeled rows as confirmed negatives. Shared observed/fitted charts still
compare the supplied PU labels to predictions; decide whether that comparison
is useful for your task. Set `common=replace(common, observed_fitted=False)`
to omit those charts.

## Run the chosen recipe

Prepare a `data` pool with targets appropriate to the selected task. The
[data example](api/data.md#prepare-a-pool) works for regression, frequency, and
the severity recipe above. Configure MLflow, then run:

<!-- example: configuration-run -->

```python
import mlflow

from ml_toolkit import train_model

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("ml-toolkit-task-recipes")
result = train_model(
    data=data,
    params=params,
    model_factory=model_factory,
    metric=metric,
    run_name="task-example",
    config=config,
)
print({name: round(value, 4) for name, value in result.validation.mean_metrics.items()})
assert result.test_metrics is not None
print({name: round(value, 4) for name, value in result.test_metrics.items()})
```

<!-- output: configuration-run -->
```text
{'mean_poisson_deviance': 0.9547, 'd2_poisson_score': -0.0823, 'mean_absolute_error': 0.707, 'root_mean_squared_error': 0.8303, 'prediction_over_observed': 0.9742, 'lift_at_10': 0.3333, 'gain_at_10': 0.0333, 'normalized_gini': -0.5079}
{'mean_poisson_deviance': 0.733, 'd2_poisson_score': -0.0916, 'mean_absolute_error': 0.631, 'root_mean_squared_error': 0.7895, 'prediction_over_observed': 0.7903, 'lift_at_10': 1.6667, 'gain_at_10': 0.1667, 'normalized_gini': 0.2}
```
<!-- /output -->

The output shown below the example uses the Frequency recipe and the data
example. Choose your task block before running this shared call; its metric
names and values will differ for other tasks.

The call creates a local `mlflow.db` and logs metrics, configuration, tables,
and figures. See the [training guide](api/training.md) to add feature selection,
an Optuna search space, or a custom evaluator. Pass those callbacks as explicit
arguments to `train_model`, alongside their configuration.

## Switching validation strategies

<!-- example: configuration-splits -->

```python
holdout_config = replace(regression_config, validation=HoldoutConfig(val_size=0.2))
no_test_config = replace(holdout_config, test_size=None)
date_test_config = replace(holdout_config, test_size="2025-01-20")
```

Run the regression configuration block first. `test_size=None` disables test;
`result.test_metrics` is then `None`. A date or `YYYY-MM-DD` string starts test
on that calendar day, inclusive, in the timezone of `data.time`. If that day is
missing, the next observation starts test. Both partitions must be nonempty;
exact datetimes are not accepted.

## Shared PSI and observed/fitted diagnostics

Configure both diagnostics in `plots.common`. They share levels fitted on the
training sample: numerical bins, categories, missing/error sentinels, and valid
bounds. The following example replaces the regression reports:

<!-- example: configuration-features -->

```python
feature_common = CommonPlotsConfig(
    features=(
        FeatureLevelSpec("age", "numerical", lower_bound=0, upper_bound=120),
        FeatureLevelSpec("region", "categorical", null_level="unknown"),
    ),
    psi=True,
    observed_fitted=True,
    n_bins=8,
)
feature_config = replace(
    regression_config,
    reporting=ReportingConfig(
        diagnostics=RegressionDiagnosticsConfig(
            plots=RegressionPlotsConfig(common=feature_common),
        ),
    ),
)
```

`features=None` selects every eligible model feature; `features=()` disables
both feature-level diagnostics. Explicit names must remain after feature
selection. Raw numerical distributions are independently configured through
`numerical_distributions`; set it to `None` to disable them. SHAP defaults off;
set `shap=ShapConfig()` (imported from `ml_toolkit.config`) and install the `shap`
extra to enable it.

`psi=True` writes score and level-distribution tables, a PSI score figure, and
monthly level distributions. `observed_fitted=True` writes one figure per feature
with weighted train and validation/test means and volume bars. Missing or
zero-weight groups have no mean and break the curve. Each validation split fits
its own levels; test uses the final training sample as reference.

Both flags default to `True` and work independently for every task. The separate
`monthly_observed_fitted` flag controls monthly target/prediction charts. Setting
the enclosing `plots=None` disables all built-in figures and PSI tables.
