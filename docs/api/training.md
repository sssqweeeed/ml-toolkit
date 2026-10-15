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

result = train_model(
    data=data,
    params={
        "iterations": 20,
        "depth": 2,
        "loss_function": "Poisson",
        "random_seed": 42,
        "thread_count": 2,
        "verbose": False,
        "allow_writing_files": False,
    },
    model_factory=CatBoostLogLinkRegressionModel,
    metric=metrics_frequency,
    run_name="frequency-holdout",
    config=TrainingConfig(
        validation=HoldoutConfig(val_size=0.25),
        test_size=0.2,
        tracking=TrackingConfig(log_final_model=False),
    ),
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
[configuration by task](../configuration.md) for selection and tuning recipes;
selection also needs `feature_ranker`, and tuning needs `param_space`.

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
