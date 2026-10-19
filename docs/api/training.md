# Training

Use `train_model` when you want chronological validation, optional selection and
tuning, a final fit, and MLflow reporting in one call.

## Run a holdout experiment

Continue with the `data` pool from the [data example](data.md#prepare-a-pool).
The [getting started](../getting-started.md) page contains a standalone version
with a larger synthetic dataset.

```python
import mlflow

from ml_toolkit import train_model
from ml_toolkit.config import HoldoutConfig, TrackingConfig, TrainingConfig
from ml_toolkit.metrics import metrics_frequency
from ml_toolkit.models import CatBoostLogLinkRegressionModel

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("ml-toolkit-examples")

params = {
    "iterations": 20,
    "depth": 2,
    "loss_function": "Poisson",
    "random_seed": 42,
    "thread_count": 2,
    "verbose": False,
    "allow_writing_files": False,
}
config = TrainingConfig(
    validation=HoldoutConfig(val_size=0.25),
    test_size=0.2,
    tracking=TrackingConfig(log_final_model=False),
)
result = train_model(
    data=data,
    params=params,
    model_factory=CatBoostLogLinkRegressionModel,
    metric=metrics_frequency,
    run_name="frequency-holdout",
    config=config,
)
print(result.validation.mean_metrics)
print(result.test_metrics)
```

Of the 24 rows, the latest 5 become test. Of the remaining 19, the latest 5
become validation and the first 14 become training. Fractions round upward.
After validation, the final model fits all 19 non-test rows and evaluates the
5 test rows. The call creates a local `mlflow.db` and logs metrics and
configuration; model packaging is disabled in this example.

## Use the result

`result.validation.mean_metrics` summarizes held-out validation scores;
`result.test_metrics` describes the future test tail. `result.model` is the final
fitted adapter. A single holdout produces one entry in
`result.validation.splits`.

To evaluate several chronological windows, change `validation` to
`TimeSeriesCVConfig(n_splits=3)`. Each window performs its own selection and
tuning when enabled. Final fitting uses the last window's settings. See
[configuration by task](../configuration.md) for reporting and task-specific
settings.

## Select features and tune parameters

Continue after the holdout example. Selection needs a ranker and tuning needs
an Optuna search-space callback. This small example visits the two-feature and
one-feature subsets, then compares two parameter trials on the selected subset.

```python
from dataclasses import replace

from optuna import Trial

from ml_toolkit.config import RFEFeatureSelectionConfig, TuningConfig
from ml_toolkit.training import importance_ranker
from ml_toolkit.types import ModelParams


def param_space(trial: Trial) -> ModelParams:
    return {"depth": trial.suggest_int("depth", 1, 3)}


search_config = replace(
    config,
    feature_selection=RFEFeatureSelectionConfig(
        min_features=1,
        metric="mean_poisson_deviance",
        direction="minimize",
        max_fits=2,
    ),
    tuning=TuningConfig(
        metric="mean_poisson_deviance",
        direction="minimize",
        n_trials=2,
        seed=42,
    ),
)
searched = train_model(
    data=data,
    params=params,
    model_factory=CatBoostLogLinkRegressionModel,
    metric=metrics_frequency,
    run_name="frequency-search",
    config=search_config,
    feature_ranker=importance_ranker,
    param_space=param_space,
)
print(searched.features)
print(searched.feature_selection)
print(searched.tuning)
```

The metric name must be a key returned by `metric(...).to_dict()`. Selection
and tuning use validation rows, so their best scores are search results. Use
`searched.test_metrics` for the independent future test evaluation. Each CV
window repeats this process; the final fit reuses the last window's settings.
`ShapRanker()` is an alternative ranker using native tree SHAP contributions;
it does not require the `shap` extra.

For an explicit train/validation pair, use `split_holdout` and
`select_features` directly. Selection returns history and column names; apply
them with `data.select_features(selection.features)` before your own fit.
`tune_optuna` also works independently when supplied a scalar `score` callback.

## Add custom evaluation

Custom evaluation needs aligned IDs in `Pool.object_id`. Continue with `data`,
`params`, and `config` from the holdout example. The evaluator below reports
unweighted MAE and returns no images; the inherited image writer does nothing.

```python
import numpy as np
import polars as pl

from ml_toolkit import Pool
from ml_toolkit.training import DatasetSplit, PostFitEvaluator
from ml_toolkit.types import FloatArray


class ExtraMetrics(PostFitEvaluator):
    def evaluate(
        self,
        *,
        object_id: pl.Series,
        y_pred: FloatArray,
        y_true: FloatArray,
        split: DatasetSplit,
    ) -> dict[str, float]:
        return {"mae": float(np.mean(np.abs(y_true - y_pred)))}


identified = Pool(
    X=data.X,
    y=data.y,
    time=data.time,
    weight=data.weight,
    cat_features=data.cat_features,
    numerical_features=data.numerical_features,
    object_id=pl.Series("object_id", range(len(data))),
)
evaluated = train_model(
    data=identified,
    params=params,
    model_factory=CatBoostLogLinkRegressionModel,
    metric=metrics_frequency,
    run_name="frequency-custom-evaluation",
    config=config,
    post_fit_evaluator=ExtraMetrics(),
)
```

The default policy evaluates available train, validation, and test rows during
validation and final fitting. Custom scores appear in MLflow under
`<split>/custom/mae`; they do not enter validation averages or the tuning
objective. The same evaluator instance is reused, so return scores for the
supplied rows and keep inputs unchanged.

Use `ReportingConfig(post_fit=PostFitEvaluationConfig(...))` to choose stages
and splits. To log images, override `write_artifacts`, save them inside the
supplied temporary `artifact_dir`, and return their paths. The method's reference
below describes supported files and directory lifetime.

## API reference

Import these operations and result types from `ml_toolkit.training`.
`train_model` and `TrainingResult` are also available from `ml_toolkit`.

::: ml_toolkit.training
    options:
      members:
        - train_model
        - select_features
        - importance_ranker
        - ShapRanker
        - tune_optuna
        - split_holdout
        - make_validation_splits
        - TrainValSplit
        - TrainingResult
        - ValidationResult
        - SplitResult
        - FeatureSelectionResult
        - FeatureSelectionStep
        - TuningResult
        - FeatureRanker
        - PostFitEvaluator
        - DatasetSplit

### Call a feature ranker

Rankers are called as `ranker(model, data)`. Their callable interfaces describe
the required feature coverage and the returned importance values.

::: ml_toolkit.training.FeatureRanker.__call__

::: ml_toolkit.training.ShapRanker.__call__
