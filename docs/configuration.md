# Configuration by task

Configurations are immutable, keyword-only dataclasses. Start with shared
settings, then choose a model, metric, and diagnostic family for your target.
Each task includes a compact recipe using `base_config` and a standalone full
configuration with feature selection, parameter tuning, SHAP, and final-model
packaging enabled. The full recipes spell out the nested settings and the
callbacks passed separately to training.
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

## Prepare source data

Run from an installed repository checkout after `python -m examples.prepare_data`.
The recipes below load cached source observations explicitly; they never invent
timestamps or relabel confirmed negatives as unlabeled records.

| Recipe | Observations and target | Evaluation |
| --- | --- | --- |
| Regression, frequency | All 17,379 UCI Bike Sharing hours; recorded rentals per hour | Chronological CV followed by the last 20% of hours |
| Binary classification | The same hours; high demand means at least 100 recorded rentals | The same chronological partitions; target counts are excluded from inputs |
| Severity | Fixed 10% sample of complete UCI Online Retail invoices; positive unit price in GBP | Train before September 2011, validate in September, test in October–November |
| Positive–unlabeled | Caller-supplied known-positive and unlabeled observations | Requires genuine timestamps and an externally justified population prior |

The [Bike setup](api/models.md#prepare-real-observations) documents its features
and units. The [retail setup](api/inflation.md#prepare-recorded-retail-prices)
documents sampling and exclusions. Retail prices illustrate positive monetary
regression; they are not insurance claim severities. The PU blocks illustrate
configuration only and are not run on either of these supervised targets.

## Imports and shared behavior

Run this block first when using a compact recipe. Full configuration blocks
include their own imports and do not require this setup. Choose either version
of one task; both define `config`, `model_factory`, `params`, and `metric` for
the same training call. Full recipes also define `feature_selector` and
`param_space`; compact recipes set these to `None`.

Full recipes require the `shap` extra (`pip install 'ml-toolkit[shap]'`) and
at least two input features to exercise elimination. Each validation split
compares up to two RFE subsets, then runs two Optuna trials on the selected
features. These small budgets keep the examples quick; increase `max_fits`,
`n_trials`, and the model's `iterations` for your experiment. `min_features=1`
is a lower bound, not a promise to select exactly one feature.

Feature selection is configured in `RFESelector` and passed as
`feature_selector`; it is not a `TrainingConfig` field. `TuningConfig` supplies
the search budget and objective, while the separate `param_space` callback
supplies the parameters to search. Both selection and tuning use the named
score returned by the task's metric. Final fitting reuses the last split's
selected features and settings; test rows never participate in either search.

| Capability | Compact recipe | Full recipe |
| --- | --- | --- |
| Feature selection | All input features | RFE with native feature importance |
| Parameter tuning | Off | Per-split Optuna search for depth and learning rate |
| SHAP | Off | Importance and beeswarm on validation/test |
| Final-model packaging | Off | Saved in MLflow with the selected input schema |
| Custom evaluation | No callback | No callback; see [custom evaluation](api/training.md#add-custom-evaluation) |

Custom evaluation uses a named project-specific `PredictionEvaluator`, or a list
of evaluators with names unique without regard to case.
`ReportingConfig.prediction_evaluation` defaults to `None`: supplying an evaluator activates
the standard policy on validation and final stages, while omitting the evaluator
disables custom evaluation. Pass `PredictionEvaluationConfig` only to override this
policy; empty stages or splits disable calls. Aligned `data.object_id` is required
before training only when the selected stages and available splits permit a call.
An empty list also disables evaluation. Each evaluator stores artifacts under
`<name>/<split>/` and metrics under `<split>/<name>/`.
Both recipes leave `prediction_evaluators=None`. Built-in metrics, PSI, and figures
need no custom evaluator. See [selector alternatives](api/training.md#choose-a-selector)
for Top-K, forward selection, and SHAP ranking.

Diagnostic split indices are checked against the validation strategy only when
diagnostics and their plots are enabled. Disabling either lets you reuse those
indices with another strategy; indices must still be positive and unique.

<!-- example: configuration-shared -->

```python
from dataclasses import replace

from ml_toolkit.config import (
    ClassificationDiagnosticsConfig,
    ClassificationPlotsConfig,
    CommonPlotsConfig,
    FeatureLevelConfig,
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

The compact recipes keep feature selection, tuning, SHAP, and model packaging off.
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

The example labels an observed hour as high demand when its recorded rental
count is at least 100. This fixed operational threshold is chosen before
splitting; the count and its components are excluded from features. Include
both classes in evaluated splits for finite ROC AUC; single-class splits return NaN.

### Using the base configuration

<!-- example: configuration-classification -->

```python
from examples.gallery_data import load_bike_pool
import polars as pl

data = load_bike_pool(download_if_missing=False)
data = data.with_target((data.y >= 100).cast(pl.Float64))

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
feature_selector = None
param_space = None
prediction_evaluators = None
```

### Full configuration

<!-- example: configuration-classification-full -->

```python
from examples.gallery_data import load_bike_pool
import polars as pl

data = load_bike_pool(download_if_missing=False)
data = data.with_target((data.y >= 100).cast(pl.Float64))

from optuna import Trial

from ml_toolkit.config import (
    ClassificationDiagnosticsConfig,
    ClassificationPlotsConfig,
    CommonPlotsConfig,
    FigureConfig,
    NumericalFeatureDistributionConfig,
    ReportingConfig,
    ShapConfig,
    TimeSeriesCVConfig,
    TrackingConfig,
    TrainingConfig,
    TuningConfig,
)
from ml_toolkit.feature_selection import RFESelector
from ml_toolkit.metrics import metrics_classification
from ml_toolkit.models import CatBoostClassificationModel
from ml_toolkit.training import importance_ranker
from ml_toolkit.types import ModelParams

classification_config = TrainingConfig(
    validation=TimeSeriesCVConfig(
        n_splits=3,
        val_size=None,
        gap=0,
        max_train_size=None,
        skip_splits=0,
    ),
    test_size=0.2,
    tuning=TuningConfig(
        metric="log_loss",
        direction="minimize",
        n_trials=2,
        timeout_seconds=None,
        seed=42,
    ),
    reporting=ReportingConfig(
        diagnostics=ClassificationDiagnosticsConfig(
            plots=ClassificationPlotsConfig(
                common=CommonPlotsConfig(
                    figure=FigureConfig(
                        figsize=None,
                        dpi=150,
                        format="png",
                    ),
                    features=None,
                    psi=True,
                    observed_fitted=True,
                    period="month",
                    n_bins=10,
                    epsilon=1e-6,
                    numerical_distributions=NumericalFeatureDistributionConfig(
                        features=None,
                        quantile_bands=((0.05, 0.95), (0.25, 0.75)),
                        show_mean=True,
                        show_median=True,
                    ),
                    shap=ShapConfig(
                        max_samples=500,
                        seed=42,
                        max_display=10,
                        dependence_features=(),
                    ),
                ),
                precision_recall=True,
                f1_threshold=True,
                calibration=True,
                gains=True,
                lift=True,
                time_observed_fitted=True,
                max_curve_points=1000,
                calibration_bins=10,
                histogram_bins=30,
            ),
        ),
        validation_splits="last",
    ),
    tracking=TrackingConfig(
        log_trial_models=False,
        log_validation_models=False,
        log_final_model=True,
        input_example_rows=5,
        model_name="model",
        registered_model_name=None,
        conda_env=None,
    ),
)
config = classification_config
model_factory = CatBoostClassificationModel
params = {
    "iterations": 20,
    "depth": 2,
    "random_seed": 42,
    "thread_count": 2,
    "verbose": False,
    "allow_writing_files": False,
    "loss_function": "Logloss",
}
metric = metrics_classification
feature_selector = RFESelector(
    min_features=1,
    metric="log_loss",
    direction="minimize",
    ranker=importance_ranker,
    max_fits=2,
    score_tolerance=0.0,
    protected_features=(),
)


def param_space(trial: Trial) -> ModelParams:
    return {
        "depth": trial.suggest_int("depth", 1, 3),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.1, log=True),
    }


prediction_evaluators = None
```

The reports include precision-recall, F1, calibration, gains/lift, temporal means,
and the shared feature diagnostics.

## Regression

Ordinary regression supports signed targets. Define a three-argument metric
returning named scores; this example uses exposure-weighted absolute error.

### Using the base configuration

<!-- example: configuration-regression -->

```python
from examples.gallery_data import load_bike_pool

data = load_bike_pool(download_if_missing=False)

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
feature_selector = None
param_space = None
prediction_evaluators = None
```

### Full configuration

<!-- example: configuration-regression-full -->

```python
from examples.gallery_data import load_bike_pool

data = load_bike_pool(download_if_missing=False)

from optuna import Trial

from ml_toolkit.config import (
    CommonPlotsConfig,
    FigureConfig,
    NumericalFeatureDistributionConfig,
    RegressionDiagnosticsConfig,
    RegressionPlotsConfig,
    ReportingConfig,
    ShapConfig,
    TimeSeriesCVConfig,
    TrackingConfig,
    TrainingConfig,
    TuningConfig,
)
from ml_toolkit.feature_selection import RFESelector
from ml_toolkit.metrics import MetricScores
from ml_toolkit.models import CatBoostRegressionModel
from ml_toolkit.training import importance_ranker
from ml_toolkit.types import FloatArray, ModelParams


def weighted_mae(
    observed: FloatArray, prediction: FloatArray, weight: FloatArray
) -> MetricScores:
    return MetricScores(
        {"mae": float((abs(observed - prediction) * weight).sum() / weight.sum())}
    )


regression_config = TrainingConfig(
    validation=TimeSeriesCVConfig(
        n_splits=3,
        val_size=None,
        gap=0,
        max_train_size=None,
        skip_splits=0,
    ),
    test_size=0.2,
    tuning=TuningConfig(
        metric="mae",
        direction="minimize",
        n_trials=2,
        timeout_seconds=None,
        seed=42,
    ),
    reporting=ReportingConfig(
        diagnostics=RegressionDiagnosticsConfig(
            task="regression",
            plots=RegressionPlotsConfig(
                common=CommonPlotsConfig(
                    figure=FigureConfig(
                        figsize=None,
                        dpi=150,
                        format="png",
                    ),
                    features=None,
                    psi=True,
                    observed_fitted=True,
                    period="month",
                    n_bins=10,
                    epsilon=1e-6,
                    numerical_distributions=NumericalFeatureDistributionConfig(
                        features=None,
                        quantile_bands=((0.05, 0.95), (0.25, 0.75)),
                        show_mean=True,
                        show_median=True,
                    ),
                    shap=ShapConfig(
                        max_samples=500,
                        seed=42,
                        max_display=10,
                        dependence_features=(),
                    ),
                ),
                gains=False,
                lift=False,
                time_observed_fitted=True,
            ),
        ),
        validation_splits="last",
    ),
    tracking=TrackingConfig(
        log_trial_models=False,
        log_validation_models=False,
        log_final_model=True,
        input_example_rows=5,
        model_name="model",
        registered_model_name=None,
        conda_env=None,
    ),
)
config = regression_config
model_factory = CatBoostRegressionModel
params = {
    "iterations": 20,
    "depth": 2,
    "random_seed": 42,
    "thread_count": 2,
    "verbose": False,
    "allow_writing_files": False,
    "loss_function": "RMSE",
}
metric = weighted_mae
feature_selector = RFESelector(
    min_features=1,
    metric="mae",
    direction="minimize",
    ranker=importance_ranker,
    max_fits=2,
    score_tolerance=0.0,
    protected_features=(),
)


def param_space(trial: Trial) -> ModelParams:
    return {
        "depth": trial.suggest_int("depth", 1, 3),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.1, log=True),
    }


prediction_evaluators = None
```

Gains and lift stay off because they require nonnegative targets with a positive
weighted total. Shared feature diagnostics and time observed/fitted remain on.

## Frequency

Use nonnegative targets with a positive weighted total in every evaluated split.
The log-link adapter fits a training-only baseline and returns positive rates.

### Using the base configuration

<!-- example: configuration-frequency -->

```python
from examples.gallery_data import load_bike_pool

data = load_bike_pool(download_if_missing=False)

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
feature_selector = None
param_space = None
prediction_evaluators = None
```

### Full configuration

<!-- example: configuration-frequency-full -->

```python
from examples.gallery_data import load_bike_pool

data = load_bike_pool(download_if_missing=False)

from optuna import Trial

from ml_toolkit.config import (
    CommonPlotsConfig,
    FigureConfig,
    NumericalFeatureDistributionConfig,
    RegressionDiagnosticsConfig,
    RegressionPlotsConfig,
    ReportingConfig,
    ShapConfig,
    TimeSeriesCVConfig,
    TrackingConfig,
    TrainingConfig,
    TuningConfig,
)
from ml_toolkit.feature_selection import RFESelector
from ml_toolkit.metrics import metrics_frequency
from ml_toolkit.models import CatBoostLogLinkRegressionModel
from ml_toolkit.training import importance_ranker
from ml_toolkit.types import ModelParams

frequency_config = TrainingConfig(
    validation=TimeSeriesCVConfig(
        n_splits=3,
        val_size=None,
        gap=0,
        max_train_size=None,
        skip_splits=0,
    ),
    test_size=0.2,
    tuning=TuningConfig(
        metric="mean_poisson_deviance",
        direction="minimize",
        n_trials=2,
        timeout_seconds=None,
        seed=42,
    ),
    reporting=ReportingConfig(
        diagnostics=RegressionDiagnosticsConfig(
            task="frequency",
            plots=RegressionPlotsConfig(
                common=CommonPlotsConfig(
                    figure=FigureConfig(
                        figsize=None,
                        dpi=150,
                        format="png",
                    ),
                    features=None,
                    psi=True,
                    observed_fitted=True,
                    period="month",
                    n_bins=10,
                    epsilon=1e-6,
                    numerical_distributions=NumericalFeatureDistributionConfig(
                        features=None,
                        quantile_bands=((0.05, 0.95), (0.25, 0.75)),
                        show_mean=True,
                        show_median=True,
                    ),
                    shap=ShapConfig(
                        max_samples=500,
                        seed=42,
                        max_display=10,
                        dependence_features=(),
                    ),
                ),
                gains=True,
                lift=True,
                time_observed_fitted=True,
            ),
        ),
        validation_splits="last",
    ),
    tracking=TrackingConfig(
        log_trial_models=False,
        log_validation_models=False,
        log_final_model=True,
        input_example_rows=5,
        model_name="model",
        registered_model_name=None,
        conda_env=None,
    ),
)
config = frequency_config
model_factory = CatBoostLogLinkRegressionModel
params = {
    "iterations": 20,
    "depth": 2,
    "random_seed": 42,
    "thread_count": 2,
    "verbose": False,
    "allow_writing_files": False,
    "loss_function": "Poisson",
}
metric = metrics_frequency
feature_selector = RFESelector(
    min_features=1,
    metric="mean_poisson_deviance",
    direction="minimize",
    ranker=importance_ranker,
    max_fits=2,
    score_tolerance=0.0,
    protected_features=(),
)


def param_space(trial: Trial) -> ModelParams:
    return {
        "depth": trial.suggest_int("depth", 1, 3),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.1, log=True),
    }


prediction_evaluators = None
```

## Severity

Use the recorded positive retail unit prices as monetary targets, with
`stock_code`, `country` and `quantity` as inputs. The test starts on 1 October
2011; the holdout uses September for validation and earlier invoices for
training. The validation row count comes from the canonical calendar split,
so whole invoices stay together. Shared feature diagnostics focus on
quantity rather than drawing thousands of product categories.

Use a Tweedie power consistent with the model loss and metric. Here `power=1.5`
allows zero targets and requires positive predictions and positive weighted
observed totals. Gamma deviance (`power=2`) requires strictly positive targets.

For strictly positive severity, set `loss_function="Gamma"` on the same
`CatBoostLogLinkRegressionModel` and set the reporting metric's `power=2.0`.
This trains the exact Gamma objective rather than approximating it with Tweedie
near power 2. See [Gamma regression](api/models.md#gamma-regression) for CPU/GPU
configuration, predictions, validation requirements and saving models.

### Using the base configuration

<!-- example: configuration-severity -->

```python
from examples.retail_data import load_retail_pool, load_retail_splits

data = load_retail_pool(download_if_missing=False)
retail_splits = load_retail_splits(download_if_missing=False)

def severity_metric(
    observed: FloatArray, prediction: FloatArray, weight: FloatArray
) -> MetricsSeverity:
    return metrics_severity(observed, prediction, weight, power=1.5)


severity_config = replace(
    base_config,
    validation=HoldoutConfig(val_size=len(retail_splits.validation)),    test_size="2011-10-01",
    reporting=ReportingConfig(
        diagnostics=RegressionDiagnosticsConfig(
            task="severity",
            plots=RegressionPlotsConfig(common=replace(common, features=(FeatureLevelConfig("quantity", "numerical"),)),
                gains=True, lift=True),
        ),
    ),
)
config = severity_config
model_factory = CatBoostLogLinkRegressionModel
params = {**base_params, "loss_function": "Tweedie:variance_power=1.5"}
metric = severity_metric
feature_selector = None
param_space = None
prediction_evaluators = None
```

### Full configuration

<!-- example: configuration-severity-full -->

```python
from examples.retail_data import load_retail_pool, load_retail_splits

data = load_retail_pool(download_if_missing=False)
retail_splits = load_retail_splits(download_if_missing=False)

from optuna import Trial

from ml_toolkit.config import (
    CommonPlotsConfig,
    FigureConfig,
    FeatureLevelConfig,
    NumericalFeatureDistributionConfig,
    RegressionDiagnosticsConfig,
    RegressionPlotsConfig,
    ReportingConfig,
    ShapConfig,
    HoldoutConfig,
    TrackingConfig,
    TrainingConfig,
    TuningConfig,
)
from ml_toolkit.feature_selection import RFESelector
from ml_toolkit.metrics import MetricsSeverity, metrics_severity
from ml_toolkit.models import CatBoostLogLinkRegressionModel
from ml_toolkit.training import importance_ranker
from ml_toolkit.types import FloatArray, ModelParams


def severity_metric(
    observed: FloatArray, prediction: FloatArray, weight: FloatArray
) -> MetricsSeverity:
    return metrics_severity(observed, prediction, weight, power=1.5)


severity_config = TrainingConfig(
    validation=HoldoutConfig(val_size=len(retail_splits.validation)),
    test_size="2011-10-01",
    tuning=TuningConfig(
        metric="mean_tweedie_deviance",
        direction="minimize",
        n_trials=2,
        timeout_seconds=None,
        seed=42,
    ),
    reporting=ReportingConfig(
        diagnostics=RegressionDiagnosticsConfig(
            task="severity",
            plots=RegressionPlotsConfig(
                common=CommonPlotsConfig(
                    figure=FigureConfig(
                        figsize=None,
                        dpi=150,
                        format="png",
                    ),
                    features=(FeatureLevelConfig("quantity", "numerical"),),
                    psi=True,
                    observed_fitted=True,
                    period="month",
                    n_bins=10,
                    epsilon=1e-6,
                    numerical_distributions=NumericalFeatureDistributionConfig(
                        features=None,
                        quantile_bands=((0.05, 0.95), (0.25, 0.75)),
                        show_mean=True,
                        show_median=True,
                    ),
                    shap=ShapConfig(
                        max_samples=500,
                        seed=42,
                        max_display=10,
                        dependence_features=(),
                    ),
                ),
                gains=True,
                lift=True,
                time_observed_fitted=True,
            ),
        ),
        validation_splits="last",
    ),
    tracking=TrackingConfig(
        log_trial_models=False,
        log_validation_models=False,
        log_final_model=True,
        input_example_rows=5,
        model_name="model",
        registered_model_name=None,
        conda_env=None,
    ),
)
config = severity_config
model_factory = CatBoostLogLinkRegressionModel
params = {
    "iterations": 20,
    "depth": 2,
    "random_seed": 42,
    "thread_count": 2,
    "verbose": False,
    "allow_writing_files": False,
    "loss_function": "Tweedie:variance_power=1.5",
}
metric = severity_metric
feature_selector = RFESelector(
    min_features=1,
    metric="mean_tweedie_deviance",
    direction="minimize",
    ranker=importance_ranker,
    max_fits=2,
    score_tolerance=0.0,
    protected_features=("quantity",),
)


def param_space(trial: Trial) -> ModelParams:
    return {
        "depth": trial.suggest_int("depth", 1, 3),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.1, log=True),
    }


prediction_evaluators = None
```

## Positive–unlabeled classification

Targets use `1` for known positives and `0` for unlabeled observations. Each
training and evaluated sample needs positive weight in both groups. The prior
is the expected positive fraction in the population, not the labeled fraction.
Use the same prior in the model and metric. The `0.2` below is an explicit
configuration assumption, not an estimate from Bike, Retail, or the proportion
of labeled examples. Replace it with a defensible population prior before
fitting. These blocks intentionally do not create `data`: provide genuine PU
observations and recorded timestamps before using the shared training call.

### Using the base configuration

<!-- example: configuration-pu -->

```python
pu_classification_config = replace(
    base_config,
    reporting=ReportingConfig(
        diagnostics=PUDiagnosticsConfig(
            plots=PUPlotsConfig(common=replace(common, observed_fitted=False)),
        ),
    ),
)
config = pu_classification_config
model_factory = CatBoostPUClassificationModel
params = {**base_params, "loss": "uPU", "pu_prior": 0.2}
metric = PUMetric(0.2, loss="uPU")
feature_selector = None
param_space = None
prediction_evaluators = None
```

### Full configuration

<!-- example: configuration-pu-full -->

```python
from optuna import Trial

from ml_toolkit.config import (
    CommonPlotsConfig,
    FigureConfig,
    NumericalFeatureDistributionConfig,
    PUDiagnosticsConfig,
    PUPlotsConfig,
    ReportingConfig,
    ShapConfig,
    TimeSeriesCVConfig,
    TrackingConfig,
    TrainingConfig,
    TuningConfig,
)
from ml_toolkit.feature_selection import RFESelector
from ml_toolkit.metrics import PUMetric
from ml_toolkit.models import CatBoostPUClassificationModel
from ml_toolkit.training import importance_ranker
from ml_toolkit.types import ModelParams

pu_classification_config = TrainingConfig(
    validation=TimeSeriesCVConfig(
        n_splits=3,
        val_size=None,
        gap=0,
        max_train_size=None,
        skip_splits=0,
    ),
    test_size=0.2,
    tuning=TuningConfig(
        metric="pu_risk",
        direction="minimize",
        n_trials=2,
        timeout_seconds=None,
        seed=42,
    ),
    reporting=ReportingConfig(
        diagnostics=PUDiagnosticsConfig(
            plots=PUPlotsConfig(
                common=CommonPlotsConfig(
                    figure=FigureConfig(
                        figsize=None,
                        dpi=150,
                        format="png",
                    ),
                    features=None,
                    psi=True,
                    observed_fitted=False,
                    period="month",
                    n_bins=10,
                    epsilon=1e-6,
                    numerical_distributions=NumericalFeatureDistributionConfig(
                        features=None,
                        quantile_bands=((0.05, 0.95), (0.25, 0.75)),
                        show_mean=True,
                        show_median=True,
                    ),
                    shap=ShapConfig(
                        max_samples=500,
                        seed=42,
                        max_display=10,
                        dependence_features=(),
                    ),
                ),
                score_distribution=True,
                positive_capture=True,
                histogram_bins=30,
            ),
        ),
        validation_splits="last",
    ),
    tracking=TrackingConfig(
        log_trial_models=False,
        log_validation_models=False,
        log_final_model=True,
        input_example_rows=5,
        model_name="model",
        registered_model_name=None,
        conda_env=None,
    ),
)
config = pu_classification_config
model_factory = CatBoostPUClassificationModel
params = {
    "iterations": 20,
    "depth": 2,
    "random_seed": 42,
    "thread_count": 2,
    "verbose": False,
    "allow_writing_files": False,
    "loss": "uPU",
    "pu_prior": 0.2,
}
metric = PUMetric(0.2, loss="uPU")
feature_selector = RFESelector(
    min_features=1,
    metric="pu_risk",
    direction="minimize",
    ranker=importance_ranker,
    max_fits=2,
    score_tolerance=0.0,
    protected_features=(),
)


def param_space(trial: Trial) -> ModelParams:
    return {
        "depth": trial.suggest_int("depth", 1, 3),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.1, log=True),
    }


prediction_evaluators = None
```

PU reports show score distributions and positive capture. They do not treat
unlabeled rows as confirmed negatives. Shared observed/fitted charts still
compare the supplied PU labels to predictions; decide whether that comparison
is useful for your task. Set `common=replace(common, observed_fitted=False)`
to omit those charts.

## Run the chosen recipe

Run exactly one compact or full recipe above: it prepares task-appropriate
`data` and settings. PU instead requires your own real PU pool and justified
prior. Then configure MLflow and call training. The captured output below uses
the compact Frequency recipe on recorded Bike observations:

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
    feature_selector=feature_selector,
    param_space=param_space,
    prediction_evaluators=prediction_evaluators,
)
print({name: round(value, 4) for name, value in result.validation.mean_metrics.items()})
assert result.test_metrics is not None
print({name: round(value, 4) for name, value in result.test_metrics.items()})
```

<!-- output: configuration-run -->
```text
{'mean_poisson_deviance': 120.7394, 'd2_poisson_score': 0.1843, 'mean_absolute_error': 111.9625, 'root_mean_squared_error': 155.9688, 'prediction_over_observed': 0.7438, 'lift_at_10': 1.6123, 'gain_at_10': 0.1612, 'normalized_gini': 0.6381}
{'mean_poisson_deviance': 159.3838, 'd2_poisson_score': 0.1984, 'mean_absolute_error': 144.2219, 'root_mean_squared_error': 205.547, 'prediction_over_observed': 0.6959, 'lift_at_10': 1.7831, 'gain_at_10': 0.1783, 'normalized_gini': 0.7351}
```
<!-- /output -->

The output shown below the example uses the compact Frequency recipe. Choose your task block before running this shared call; its metric
names and values will differ for other tasks.

The call creates a local `mlflow.db` and logs metrics, configuration, tables,
and figures. Full recipes also log RFE history, Optuna trials, SHAP figures,
and the final model. `result.features`, `result.feature_selection`, and
`result.tuning` describe the last split's selected settings. Use
`data.select_features(result.features)` when predicting with `result.model`;
the selected model expects its fitted input schema. The model saved in MLflow
can be reloaded for serving as shown in the
[packaging example](api/training.md#implement-a-selector). The
[training guide](api/training.md) explains alternative selectors, search spaces,
and custom evaluators.

## Inspect a complete run

Run the **Frequency → Full configuration** block, then this example from a
source checkout with the `shap` extra installed. It uses all 17,379 hourly records
from the [UCI Bike Sharing dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset),
with recorded dates and hours from 2011–2012, eleven weather/calendar features,
rental counts as targets, and unit weights per recorded hour. The preparation command caches
the official archive in `.cache/uci-bike-sharing`; example execution requires
that cache and does not download data. No dates, targets,
or distribution shifts are synthesized; no rows are sampled before splitting.

The full configuration reserves the latest 20% of rows for test and performs
three expanding chronological validation splits on the preceding rows.
Selection and tuning are repeated independently in each split. SHAP uses at
most 500 actual test rows, sampled with seed 42, and explains the final model's
raw log-rate output. The preview below is generated by this exact run.

<!-- example: configuration-full-run -->

```python
from pathlib import Path

import mlflow

from examples.gallery_data import load_bike_pool
from ml_toolkit import train_model
from ml_toolkit.explainability import calculate_shap_values
from ml_toolkit.plots.explainability import shap_importance
from ml_toolkit.training import split_holdout
from ml_toolkit.config import HoldoutConfig

data = load_bike_pool(download_if_missing=False)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("ml-toolkit-task-recipes")
result = train_model(
    data=data,
    params=params,
    model_factory=model_factory,
    metric=metric,
    run_name="frequency-full-example",
    config=config,
    feature_selector=feature_selector,
    param_space=param_space,
    prediction_evaluators=prediction_evaluators,
)
assert result.feature_selection is not None
assert result.tuning is not None
assert result.test_metrics is not None
print("Selected features:", result.features)
print("Visited subset sizes:", [
    len(step.features) for step in result.feature_selection.steps
])
print("Best trial:", result.tuning.best_trial)
print("Test Poisson deviance:", round(result.test_metrics["mean_poisson_deviance"], 4))

test = split_holdout(data, HoldoutConfig(val_size=config.test_size)).val
selected_test = test.select_features(result.features)
shap_config = config.reporting.diagnostics.plots.common.shap
explanation = calculate_shap_values(
    result.model.native_model, selected_test.X, config=shap_config
)
shap_importance(explanation, Path("frequency-full-shap.png"), config=shap_config)
```

<!-- output: configuration-full-run -->
```text
Selected features: ('temperature', 'feels_like_temperature', 'humidity', 'wind_speed', 'hour', 'working_day', 'weekday', 'holiday', 'season', 'weather', 'month')
Visited subset sizes: [11, 1]
Best trial: 0
Test Poisson deviance: 114.3487
```
<!-- /output -->

![Test SHAP importance from the full frequency recipe](assets/plots/frequency-full-shap.png)

`steps` contains every subset evaluated in the last RFE search. A winning subset
may contain more than `min_features`; RFE retains the best validation score,
preferring a smaller subset only within `score_tolerance`. `best_trial` is
zero-based. This small search demonstrates the workflow, not a tuned benchmark.
The test deviance describes only the untouched future tail.

The MLflow experiment contains all split histories under `selection/history.json`,
per-trial runs under each split's `tuning` run, final diagnostics under
`final_fit`, and the packaged final model with its selected input schema.
SHAP importance and beeswarm figures are under `figures/shap/val/` and
`figures/shap/test/` in the corresponding runs. `frequency-full-shap.png` is a
local preview of the final test importance plot.

Regenerate the captured output and preview with
`python -m examples.render_examples --page docs/configuration.md`; add `--check`
to execute the examples and detect stale output or a missing/changed figure.

## Switching validation strategies

<!-- example: configuration-splits -->

```python
from dataclasses import replace

from ml_toolkit.config import HoldoutConfig

holdout_config = replace(regression_config, validation=HoldoutConfig(val_size=0.2))
no_test_config = replace(holdout_config, test_size=None)
date_test_config = replace(holdout_config, test_size="2012-10-01")
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
from dataclasses import replace

from ml_toolkit.config import FeatureLevelConfig

feature_common = CommonPlotsConfig(
    features=(
        FeatureLevelConfig("humidity", "numerical", lower_bound=0, upper_bound=1),
        FeatureLevelConfig("weather", "categorical", null_level="unknown"),
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
`numerical_distributions`; set it to `None` to disable them. SHAP defaults off
in the compact recipes; the full recipes enable it with `ShapConfig`.
Set `shap=None` to disable it. Empty `dependence_features` disables
only dependence plots; importance and beeswarm remain enabled. If you name a
dependence feature, keep it in `protected_features` so selection cannot remove
it. The same constraint applies to explicit diagnostic feature lists.

`psi=True` writes score and level-distribution tables, a PSI score figure, and
time-based level distributions. `observed_fitted=True` writes one figure per feature
with weighted train and validation/test means and volume bars. Missing or
zero-weight groups have no mean and break the curve. Each validation split fits
its own levels; test uses the final training sample as reference.

Within each MLflow run, train-versus-validation/test comparisons are grouped under
`figures/observed_fitted/`, `figures/feature_distributions/`, and `figures/psi/`,
with PSI tables under `tables/psi/`. These shared artifacts have no `val` or `test`
subfolder. Performance and SHAP figures describe only the evaluated dataset and
retain that subfolder under `figures/performance/` and `figures/shap/`.

Both flags default to `True` and work independently for every task. The separate
`time_observed_fitted` flag controls time-based target/prediction charts. Setting
the enclosing `plots=None` disables all built-in figures and PSI tables.

Set `CommonPlotsConfig(period="week")` to group all temporal diagnostics by week.
The default is `"month"`; `"minute"`, `"hour"`, `"day"`, `"quarter"`, and `"year"`
are also supported. Week boundaries are Mondays, and all calendar boundaries
use the data's timezone. Minute/hour grouping requires `Datetime` inputs.
Independent PSI analysis accepts the same option in `PSIConfig(period=...)`.
Temporal diagnostics require the latest reference timestamp to be at or before
the earliest current timestamp. Equal endpoints are allowed; overlap beyond
that endpoint or reversed ranges raise `ValueError`. Splitting inside a grouping
period is allowed.
