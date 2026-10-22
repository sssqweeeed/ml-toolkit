# Configuration by task

Configurations are immutable, keyword-only dataclasses. Start with shared
settings, then choose a model, metric, and report objects for your target.
Each task includes a compact recipe using `base_config` and a standalone full
configuration with feature selection, parameter tuning, SHAP, and final-model
packaging enabled. The full recipes spell out the shared settings and the
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
The recipes load cached data; PU training requires a caller-supplied pool.

| Recipe | Observations and target | Evaluation |
| --- | --- | --- |
| Regression, frequency | All 17,379 UCI Bike Sharing hours; recorded rentals per hour | Chronological CV followed by the last 20% of hours |
| Binary classification | The same hours; high demand means at least 100 recorded rentals | The same chronological partitions; target counts are excluded from inputs |
| Severity | Fixed 10% sample of complete UCI Online Retail invoices; positive unit price in GBP | Train before September 2011, validate in September, test in October–November |
| Positive–unlabeled | Caller-supplied known-positive and unlabeled observations | Requires genuine timestamps and an externally justified population prior |

The [Bike setup](api/models.md#prepare-real-observations) documents its features
and units. The [retail setup](api/inflation.md#prepare-recorded-retail-prices)
documents sampling and exclusions for the monetary regression example.

## Imports and shared behavior

Run this block first when using a compact recipe. Full configuration blocks
include their own imports and do not require this setup. Choose either version
of one task; both define `config`, `model_factory`, `params`, and `metric` for
the same training call, together with a data-bound `splitter`. Full recipes also define `feature_selector` and
`param_space`; compact recipes set these to `None`.

Full recipes require the `shap` extra (`pip install 'ml-toolkit[shap]'`) and
at least two input features to exercise elimination. Each validation split
compares up to two RFE subsets, then runs two Optuna trials on the selected
features. These small budgets keep the examples quick; increase `max_fits`,
`n_trials`, and the model's `iterations` for your experiment. `min_features=1`
is a lower bound, not a promise to select exactly one feature.

Pass an `RFESelector` as `feature_selector`. `TuningConfig` supplies the search
budget and objective; `param_space` supplies the parameters to search. Both use
a named score returned by the task's metric.

| Capability | Compact recipe | Full recipe |
| --- | --- | --- |
| Feature selection | All input features | RFE with native feature importance |
| Parameter tuning | Off | Per-split Optuna search for depth and learning rate |
| SHAP | Off | Importance and beeswarm on validation/test |
| Final-model packaging | Off | Saved in MLflow with the selected input schema |
| Reports | Explicit evaluator list | Explicit evaluator list plus SHAP |

Pass named report objects from `ml_toolkit.evaluation` as
`prediction_evaluators`. `ReportingConfig` supplies shared feature definitions,
figure settings, PSI and SHAP. Its `on_tuning=False`, `on_validation=True`, and
`on_final=True` flags control additional reports on each stage; primary metrics
always run. See [custom evaluation](api/training.md#add-custom-evaluation) for
the evaluator contract and metric logging.

<!-- example: configuration-shared -->

```python
from ml_toolkit.training import Holdout, TimeSeriesCV
from dataclasses import replace
from ml_toolkit.config import (
    FeatureLevelConfig,
    ReportingConfig,
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

base_config = TrainingConfig(tracking=TrackingConfig(log_final_model=False))
base_params = {
    "iterations": 20,
    "depth": 2,
    "random_seed": 42,
    "thread_count": 2,
    "verbose": False,
    "allow_writing_files": False,
}
common = ReportingConfig(psi=True)
```

- Rows must have ascending timestamps; splits never shuffle. Integers specify
  row counts, and fractions round upward. Test fractions use the original data;
  validation fractions use the retained non-test rows. `gap` and `max_train_size`
  count rows.
- Every validation split independently selects features and tunes parameters
  when enabled. Final fitting uses the last split's features, parameters,
  iteration budget, and classification threshold. It includes former validation
  and gap rows, but excludes test.

## Binary classification

The example labels an observed hour as high demand when its recorded rental
count is at least 100. This fixed operational threshold is chosen before
splitting; the count and its components are excluded from features. Include
both classes in evaluated splits for finite ROC AUC; single-class splits return NaN.

### Using the base configuration

<!-- example: configuration-classification -->

```python
from ml_toolkit.evaluation import (
    PrecisionRecall,
    F1Threshold,
    Calibration,
    Gains,
    Lift,
    ObservedFitted,
    TimeObservedFitted,
    NumericalFeatureDistributions,
)
from examples.gallery_data import load_bike_pool
import polars as pl

data = load_bike_pool(download_if_missing=False)
data = data.with_target((data.y >= 100).cast(pl.Float64))
classification_config = replace(base_config, reporting=common)
splitter = TimeSeriesCV(data, n_splits=3, test_size=0.2)
config = classification_config
model_factory = CatBoostClassificationModel
params = {**base_params, "loss_function": "Logloss"}
metric = metrics_classification
feature_selector = None
param_space = None
prediction_evaluators = [
    PrecisionRecall(),
    F1Threshold(),
    Calibration(),
    Gains(),
    Lift(),
    ObservedFitted(),
    TimeObservedFitted(),
    NumericalFeatureDistributions(),
]
```

### Full configuration

<!-- example: configuration-classification-full -->

```python
from ml_toolkit.training import TimeSeriesCV
from ml_toolkit.evaluation import (
    PrecisionRecall,
    F1Threshold,
    Calibration,
    Gains,
    Lift,
    ObservedFitted,
    TimeObservedFitted,
    NumericalFeatureDistributions,
)
from examples.gallery_data import load_bike_pool
import polars as pl

data = load_bike_pool(download_if_missing=False)
data = data.with_target((data.y >= 100).cast(pl.Float64))
from optuna import Trial
from ml_toolkit.config import (
    FigureConfig,
    ReportingConfig,
    ShapConfig,
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
    tuning=TuningConfig(
        metric="log_loss",
        direction="minimize",
        n_trials=2,
        timeout_seconds=None,
        seed=42,
    ),
    reporting=ReportingConfig(
        figure=FigureConfig(figsize=None, dpi=150, format="png"),
        features=None,
        psi=True,
        period="month",
        n_bins=10,
        epsilon=1e-06,
        shap=ShapConfig(
            max_samples=500, seed=42, max_display=10, dependence_features=()
        ),
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
splitter = TimeSeriesCV(
    data,
    n_splits=3,
    val_size=None,
    test_size=0.2,
    gap=0,
    max_train_size=None,
    skip_splits=0,
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


prediction_evaluators = [
    PrecisionRecall(),
    F1Threshold(),
    Calibration(),
    Gains(),
    Lift(),
    ObservedFitted(),
    TimeObservedFitted(),
    NumericalFeatureDistributions(),
]
```

## Regression

Ordinary regression supports signed targets. Define a three-argument metric
returning named scores; this example uses exposure-weighted absolute error.

### Using the base configuration

<!-- example: configuration-regression -->

```python
from ml_toolkit.evaluation import (
    ObservedFitted,
    TimeObservedFitted,
    NumericalFeatureDistributions,
)
from examples.gallery_data import load_bike_pool

data = load_bike_pool(download_if_missing=False)


def weighted_mae(
    observed: FloatArray, prediction: FloatArray, weight: FloatArray
) -> MetricScores:
    return MetricScores(
        {"mae": float((abs(observed - prediction) * weight).sum() / weight.sum())}
    )


regression_config = replace(base_config, reporting=common)
splitter = TimeSeriesCV(data, n_splits=3, test_size=0.2)
config = regression_config
model_factory = CatBoostRegressionModel
params = {**base_params, "loss_function": "RMSE"}
metric = weighted_mae
feature_selector = None
param_space = None
prediction_evaluators = [
    ObservedFitted(),
    TimeObservedFitted(),
    NumericalFeatureDistributions(),
]
```

### Full configuration

<!-- example: configuration-regression-full -->

```python
from ml_toolkit.training import TimeSeriesCV
from ml_toolkit.evaluation import (
    ObservedFitted,
    TimeObservedFitted,
    NumericalFeatureDistributions,
)
from examples.gallery_data import load_bike_pool

data = load_bike_pool(download_if_missing=False)
from optuna import Trial
from ml_toolkit.config import (
    FigureConfig,
    ReportingConfig,
    ShapConfig,
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
    tuning=TuningConfig(
        metric="mae", direction="minimize", n_trials=2, timeout_seconds=None, seed=42
    ),
    reporting=ReportingConfig(
        figure=FigureConfig(figsize=None, dpi=150, format="png"),
        features=None,
        psi=True,
        period="month",
        n_bins=10,
        epsilon=1e-06,
        shap=ShapConfig(
            max_samples=500, seed=42, max_display=10, dependence_features=()
        ),
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
splitter = TimeSeriesCV(
    data,
    n_splits=3,
    val_size=None,
    test_size=0.2,
    gap=0,
    max_train_size=None,
    skip_splits=0,
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


prediction_evaluators = [
    ObservedFitted(),
    TimeObservedFitted(),
    NumericalFeatureDistributions(),
]
```

Gains and lift require nonnegative targets with a positive weighted total.

## Frequency

Use nonnegative targets with a positive weighted total in every evaluated split.
The log-link adapter fits a training-only baseline and returns positive rates.

### Using the base configuration

<!-- example: configuration-frequency -->

```python
from ml_toolkit.evaluation import (
    Gains,
    Lift,
    ObservedFitted,
    TimeObservedFitted,
    NumericalFeatureDistributions,
)
from examples.gallery_data import load_bike_pool

data = load_bike_pool(download_if_missing=False)
frequency_config = replace(base_config, reporting=common)
splitter = TimeSeriesCV(data, n_splits=3, test_size=0.2)
config = frequency_config
model_factory = CatBoostLogLinkRegressionModel
params = {**base_params, "loss_function": "Poisson"}
metric = metrics_frequency
feature_selector = None
param_space = None
prediction_evaluators = [
    Gains(),
    Lift(),
    ObservedFitted(),
    TimeObservedFitted(),
    NumericalFeatureDistributions(),
]
```

### Full configuration

<!-- example: configuration-frequency-full -->

```python
from ml_toolkit.training import TimeSeriesCV
from ml_toolkit.evaluation import (
    Gains,
    Lift,
    ObservedFitted,
    TimeObservedFitted,
    NumericalFeatureDistributions,
)
from examples.gallery_data import load_bike_pool

data = load_bike_pool(download_if_missing=False)
from optuna import Trial
from ml_toolkit.config import (
    FigureConfig,
    ReportingConfig,
    ShapConfig,
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
    tuning=TuningConfig(
        metric="mean_poisson_deviance",
        direction="minimize",
        n_trials=2,
        timeout_seconds=None,
        seed=42,
    ),
    reporting=ReportingConfig(
        figure=FigureConfig(figsize=None, dpi=150, format="png"),
        features=None,
        psi=True,
        period="month",
        n_bins=10,
        epsilon=1e-06,
        shap=ShapConfig(
            max_samples=500, seed=42, max_display=10, dependence_features=()
        ),
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
splitter = TimeSeriesCV(
    data,
    n_splits=3,
    val_size=None,
    test_size=0.2,
    gap=0,
    max_train_size=None,
    skip_splits=0,
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


prediction_evaluators = [
    Gains(),
    Lift(),
    ObservedFitted(),
    TimeObservedFitted(),
    NumericalFeatureDistributions(),
]
```

## Severity

Use the recorded positive retail unit prices as monetary targets, with
`stock_code`, `country` and `quantity` as inputs. The test starts on 1 October
2011; the holdout uses September for validation and earlier invoices for
training. The validation row count comes from the canonical calendar split,
so whole invoices stay together. Feature diagnostics use quantity.

Use a Tweedie power consistent with the model loss and metric. Here `power=1.5`
allows zero targets and requires positive predictions and positive weighted
observed totals. Gamma deviance (`power=2`) requires strictly positive targets.

For strictly positive severity, set `loss_function="Gamma"` on the same
`CatBoostLogLinkRegressionModel` and set the reporting metric's `power=2.0`.
See [Gamma regression](api/models.md#gamma-regression) for CPU/GPU configuration,
predictions, validation requirements and saving models.

### Using the base configuration

<!-- example: configuration-severity -->

```python
from ml_toolkit.evaluation import (
    Gains,
    Lift,
    ObservedFitted,
    TimeObservedFitted,
    NumericalFeatureDistributions,
)
from examples.retail_data import load_retail_pool, load_retail_splits

data = load_retail_pool(download_if_missing=False)
retail_splits = load_retail_splits(download_if_missing=False)


def severity_metric(
    observed: FloatArray, prediction: FloatArray, weight: FloatArray
) -> MetricsSeverity:
    return metrics_severity(observed, prediction, weight, power=1.5)


severity_config = replace(
    base_config,
    reporting=replace(common, features=(FeatureLevelConfig("quantity", "numerical"),)),
)
splitter = Holdout(data, val_size=len(retail_splits.validation), test_size="2011-10-01")
config = severity_config
model_factory = CatBoostLogLinkRegressionModel
params = {**base_params, "loss_function": "Tweedie:variance_power=1.5"}
metric = severity_metric
feature_selector = None
param_space = None
prediction_evaluators = [
    Gains(),
    Lift(),
    ObservedFitted(),
    TimeObservedFitted(),
    NumericalFeatureDistributions(),
]
```

### Full configuration

<!-- example: configuration-severity-full -->

```python
from ml_toolkit.training import Holdout
from ml_toolkit.evaluation import (
    Gains,
    Lift,
    ObservedFitted,
    TimeObservedFitted,
    NumericalFeatureDistributions,
)
from examples.retail_data import load_retail_pool, load_retail_splits

data = load_retail_pool(download_if_missing=False)
retail_splits = load_retail_splits(download_if_missing=False)
from optuna import Trial
from ml_toolkit.config import (
    FigureConfig,
    FeatureLevelConfig,
    ReportingConfig,
    ShapConfig,
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
    tuning=TuningConfig(
        metric="mean_tweedie_deviance",
        direction="minimize",
        n_trials=2,
        timeout_seconds=None,
        seed=42,
    ),
    reporting=ReportingConfig(
        figure=FigureConfig(figsize=None, dpi=150, format="png"),
        features=(FeatureLevelConfig("quantity", "numerical"),),
        psi=True,
        period="month",
        n_bins=10,
        epsilon=1e-06,
        shap=ShapConfig(
            max_samples=500, seed=42, max_display=10, dependence_features=()
        ),
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
splitter = Holdout(
    data,
    val_size=len(retail_splits.validation),
    test_size="2011-10-01",
    gap=0,
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


prediction_evaluators = [
    Gains(),
    Lift(),
    ObservedFitted(),
    TimeObservedFitted(),
    NumericalFeatureDistributions(),
]
```

## Positive–unlabeled classification

Targets use `1` for known positives and `0` for unlabeled observations. Each
training and evaluated sample needs positive weight in both groups. The prior
is the expected positive fraction in the population, not the labeled fraction.
Use the same prior in the model and metric. Replace the example prior `0.2`
with an estimate appropriate to your population, and supply `data` containing
known-positive and unlabeled observations with recorded timestamps.

### Using the base configuration

<!-- example: configuration-pu -->

```python
from ml_toolkit.evaluation import (
    PUScoreDistribution,
    PUPositiveCapture,
    NumericalFeatureDistributions,
)

pu_classification_config = replace(base_config, reporting=replace(common))
splitter = TimeSeriesCV(data, n_splits=3, test_size=0.2)
config = pu_classification_config
model_factory = CatBoostPUClassificationModel
params = {**base_params, "loss": "uPU", "pu_prior": 0.2}
metric = PUMetric(0.2, loss="uPU")
feature_selector = None
param_space = None
prediction_evaluators = [
    PUScoreDistribution(),
    PUPositiveCapture(),
    NumericalFeatureDistributions(),
]
```

### Full configuration

<!-- example: configuration-pu-full -->

```python
from ml_toolkit.training import TimeSeriesCV
from ml_toolkit.evaluation import (
    PUScoreDistribution,
    PUPositiveCapture,
    NumericalFeatureDistributions,
)
from optuna import Trial
from ml_toolkit.config import (
    FigureConfig,
    ReportingConfig,
    ShapConfig,
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
    tuning=TuningConfig(
        metric="pu_risk",
        direction="minimize",
        n_trials=2,
        timeout_seconds=None,
        seed=42,
    ),
    reporting=ReportingConfig(
        figure=FigureConfig(figsize=None, dpi=150, format="png"),
        features=None,
        psi=True,
        period="month",
        n_bins=10,
        epsilon=1e-06,
        shap=ShapConfig(
            max_samples=500, seed=42, max_display=10, dependence_features=()
        ),
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
splitter = TimeSeriesCV(
    data,
    n_splits=3,
    val_size=None,
    test_size=0.2,
    gap=0,
    max_train_size=None,
    skip_splits=0,
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


prediction_evaluators = [
    PUScoreDistribution(),
    PUPositiveCapture(),
    NumericalFeatureDistributions(),
]
```

PU reports show score distributions and positive capture. Observed/fitted
compares supplied labels to predictions; unlabeled rows are not confirmed
negatives. Include `ObservedFitted()` only when that comparison suits the task.

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
    splitter=splitter,
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

Run **Frequency → Full configuration**, then this example to inspect selection,
tuning and SHAP results. It uses the [Bike data prepared above](#prepare-source-data)
with three chronological validation splits and the latest 20% of rows as test.
SHAP explains the final model's raw log-rate output for at most 500 test rows,
sampled with seed 42.

<!-- example: configuration-full-run -->

```python
from pathlib import Path
import mlflow
from ml_toolkit import train_model
from ml_toolkit.explainability import calculate_shap_values
from ml_toolkit.plots.explainability import shap_importance

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("ml-toolkit-task-recipes")
result = train_model(
    splitter=splitter,
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
print(
    "Visited subset sizes:",
    [len(step.features) for step in result.feature_selection.steps],
)
print("Best trial:", result.tuning.best_trial)
print("Test Poisson deviance:", round(result.test_metrics["mean_poisson_deviance"], 4))
test = splitter.get_final().heldout
assert test is not None
selected_test = test.select_features(result.features)
shap_config = config.reporting.shap
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
zero-based. Test deviance measures error on the held-out future tail.

The MLflow experiment contains all split histories under `selection/history.json`,
per-trial runs under each split's `tuning` run, final diagnostics under
`final_fit`, and the packaged final model with its selected input schema.
SHAP importance and beeswarm figures are under `shap/val/` and
`shap/test/` in the corresponding runs. `frequency-full-shap.png` is a
local preview of the final test importance plot.

Regenerate the captured output and preview with
`python -m examples.render_examples --page docs/configuration.md`; add `--check`
to execute the examples and detect stale output or a missing/changed figure.

## Switching validation strategies

<!-- example: configuration-splits -->

```python
from ml_toolkit.training import Holdout

holdout = Holdout(data, val_size=0.2, test_size=0.2)
no_test = Holdout(data, val_size=0.2)
date_test = Holdout(data, val_size=0.2, test_size="2012-10-01")
```

Run the regression configuration block first and pass the chosen object as
`splitter`. `test_size=None` disables test;
`result.test_metrics` is then `None`. A date or `YYYY-MM-DD` string starts test
on that calendar day, inclusive, in the timezone of `data.time`. If that day is
missing, the next observation starts test. Both partitions must be nonempty;
exact datetimes are not accepted.

## Shared PSI and observed/fitted diagnostics

`ReportingConfig.features` and `n_bins` define common levels for PSI and
`ObservedFitted`: quantile bins, categories, missing/error labels and bounds.
Levels are fitted on training rows.

<!-- example: configuration-features -->

```python
from dataclasses import replace
from ml_toolkit.config import FeatureLevelConfig, ReportingConfig
from ml_toolkit.evaluation import ObservedFitted, TimeObservedFitted

feature_config = replace(
    regression_config,
    reporting=ReportingConfig(
        psi=True,
        features=(
            FeatureLevelConfig("humidity", "numerical", lower_bound=0, upper_bound=1),
            FeatureLevelConfig("weather", "categorical", null_level="unknown"),
        ),
        n_bins=8,
    ),
)
feature_reports = [ObservedFitted(), TimeObservedFitted(period="week")]
```

`features=None` infers all model features; `features=()` disables level-based
PSI and observed/fitted. Explicit names must survive feature selection.
Numerical distributions use original values and their evaluator's own settings.
SHAP defaults off; use `ShapConfig` to enable it and protect any explicitly
requested dependence features from feature selection.

Comparison reports use train + val during tuning and validation, or train + test
after final fitting. Observed/fitted and numerical plots show both samples;
PSI measures their distribution differences. Train-only runs skip these reports.
Files are stored under the evaluator's name:

| Report | Default artifact paths |
| --- | --- |
| Observed/fitted | `observed_fitted/<feature>.png` |
| Temporal observed/fitted | `time_observed_fitted/time.png` |
| Numerical distributions | `numerical_feature_distributions/<feature>.png` |
| PSI | `psi/scores.png`, `psi/scores.json`, distribution tables and temporal plots |

`FigureConfig.format` selects the image format.

PSI shares count rows, observed/fitted uses weights, and numerical distributions
are unweighted. Observed/fitted means are undefined for sample/level pairs with
zero total weight.

`ReportingConfig.period` controls temporal PSI only. Each temporal evaluator
owns its `period`, defaulting to `"month"`. Minute, hour, day, week, quarter and
year are also supported; weeks begin on Monday and input timezones are kept.
Minute/hour grouping requires Datetime inputs. Reference timestamps must be at
or before current timestamps, including when a split falls inside one period.
