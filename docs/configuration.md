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

Custom evaluation uses a named project-specific `PostFitEvaluator`, or a list
of evaluators with names unique without regard to case.
`ReportingConfig.post_fit` defaults to `None`: supplying an evaluator activates
the standard policy on validation and final stages, while omitting the evaluator
disables custom evaluation. Pass `PostFitEvaluationConfig` only to override this
policy; empty stages or splits disable calls. Aligned `data.object_id` is required
before training only when the selected stages and available splits permit a call.
An empty list also disables evaluation. Each evaluator stores artifacts under
`<name>/<split>/` and metrics under `<split>/<name>/`.
Both recipes leave `post_fit_evaluator=None`. Built-in metrics, PSI, and figures
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

Use a pool whose targets contain confirmed `0` and `1` labels. Include both
classes in evaluated splits for finite ROC AUC; single-class splits return NaN.

### Using the base configuration

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
feature_selector = None
param_space = None
post_fit_evaluator = None
```

### Full configuration

<!-- example: configuration-classification-full -->

```python
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


post_fit_evaluator = None
```

The reports include precision-recall, F1, calibration, gains/lift, temporal means,
and the shared feature diagnostics.

## Regression

Ordinary regression supports signed targets. Define a three-argument metric
returning named scores; this example uses exposure-weighted absolute error.

### Using the base configuration

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
feature_selector = None
param_space = None
post_fit_evaluator = None
```

### Full configuration

<!-- example: configuration-regression-full -->

```python
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


post_fit_evaluator = None
```

Gains and lift stay off because they require nonnegative targets with a positive
weighted total. Shared feature diagnostics and time observed/fitted remain on.

## Frequency

Use nonnegative targets with a positive weighted total in every evaluated split.
The log-link adapter fits a training-only baseline and returns positive rates.

### Using the base configuration

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
feature_selector = None
param_space = None
post_fit_evaluator = None
```

### Full configuration

<!-- example: configuration-frequency-full -->

```python
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


post_fit_evaluator = None
```

## Severity

Use a Tweedie power consistent with the model loss and metric. Here `power=1.5`
allows zero targets and requires positive predictions and positive weighted
observed totals. Gamma deviance (`power=2`) requires strictly positive targets.

### Using the base configuration

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
feature_selector = None
param_space = None
post_fit_evaluator = None
```

### Full configuration

<!-- example: configuration-severity-full -->

```python
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
from ml_toolkit.metrics import MetricsSeverity, metrics_severity
from ml_toolkit.models import CatBoostLogLinkRegressionModel
from ml_toolkit.training import importance_ranker
from ml_toolkit.types import FloatArray, ModelParams


def severity_metric(
    observed: FloatArray, prediction: FloatArray, weight: FloatArray
) -> MetricsSeverity:
    return metrics_severity(observed, prediction, weight, power=1.5)


severity_config = TrainingConfig(
    validation=TimeSeriesCVConfig(
        n_splits=3,
        val_size=None,
        gap=0,
        max_train_size=None,
        skip_splits=0,
    ),
    test_size=0.2,
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
    protected_features=(),
)


def param_space(trial: Trial) -> ModelParams:
    return {
        "depth": trial.suggest_int("depth", 1, 3),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.1, log=True),
    }


post_fit_evaluator = None
```

## Positive–unlabeled classification

Targets use `1` for known positives and `0` for unlabeled observations. Each
training and evaluated sample needs positive weight in both groups. The prior
is the expected positive fraction in the population, not the labeled fraction.
Use the same prior in the model and metric.

### Using the base configuration

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
feature_selector = None
param_space = None
post_fit_evaluator = None
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


post_fit_evaluator = None
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
    feature_selector=feature_selector,
    param_space=param_space,
    post_fit_evaluator=post_fit_evaluator,
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

The output shown below the example uses the compact Frequency recipe and the data
example. Choose your task block before running this shared call; its metric
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
rental counts as targets, and unit weights per recorded hour. The loader caches
the official archive in `.cache/uci-bike-sharing` on first use. No dates, targets,
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

data = load_bike_pool()
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
    post_fit_evaluator=post_fit_evaluator,
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
from dataclasses import replace

from ml_toolkit.config import FeatureLevelSpec

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
Temporal diagnostics require the latest reference timestamp to be strictly
earlier than the earliest current timestamp. Overlap or equal endpoints raise
`ValueError`; splitting inside a grouping period is allowed.
