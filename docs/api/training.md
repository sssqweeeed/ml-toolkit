# Training

`train_model` runs validation, optional feature selection and parameter tuning,
then fits a final model and evaluates optional test data. A
[splitter](splitting.md) supplies the rows for each fit.

## Run a holdout experiment

Run this page from an installed repository checkout after
`python -m examples.prepare_data --dataset bike`. The shared
`examples.training_setup.prepare_training` loads all 17,379 original
[UCI Bike Sharing hours](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)
from 2011–2012, with eleven weather/calendar features, rental counts as targets
and unit weights. Inputs exclude count components that reveal the target.
It configures a local `mlflow.db` and disables automatic model packaging.
The model uses a Poisson loss; predictions are rentals per recorded hour.

<!-- example: training-holdout -->

```python
from examples.training_setup import prepare_training
from ml_toolkit import train_model
from ml_toolkit.metrics import metrics_frequency
from ml_toolkit.models import CatBoostLogLinkRegressionModel
from ml_toolkit.training import Holdout

data, params, config = prepare_training()
splitter = Holdout(data, val_size=0.25, test_size=0.2)
result = train_model(
    splitter=splitter,
    params=params,
    model_factory=CatBoostLogLinkRegressionModel,
    metric=metrics_frequency,
    run_name="frequency-holdout",
    config=config,
)
print(
    "Validation Poisson deviance:",
    round(result.validation.mean_metrics["mean_poisson_deviance"], 4),
)
assert result.test_metrics is not None
print(
    "Test Poisson deviance:",
    round(result.test_metrics["mean_poisson_deviance"], 4),
)
```

<!-- output: training-holdout -->
```text
Validation Poisson deviance: 118.8938
Test Poisson deviance: 111.7045
```
<!-- /output -->

The latest 20% of rows are test; the last 25% of the remaining rows are
validation. This `Holdout` uses all non-test rows for its final fit. Fractions
round upward. The workflow logs metrics and configuration to MLflow.

## Understand the run

1. Consume each validation pair in the splitter's iterator order.
2. Select columns and tune parameters when those extensions are supplied.
3. Fit the validation model and run enabled [evaluation reports](../evaluation/index.md).
4. Request `splitter.get_final()`, fit a fresh model on its `train`, and evaluate
   its optional `heldout` as test.

The last yielded validation split supplies the final features, parameters,
iteration budget and classification threshold. The splitter determines final
training rows; they need not match the last validation train. The workflow
requests final data only after selection and tuning; your splitter owns sample
separation. [Reporting stage flags](configuration.md)
control additional reports; primary metrics always run.

## Use the result

| Result | Meaning |
| --- | --- |
| `result.model` | Final fitted adapter; use `predict_pool` for prepared data |
| `result.features` | Selected columns in fitted order |
| `result.validation.splits` | Validation results in execution order |
| `result.validation.mean_metrics` | Mean validation metrics across splits |
| `result.test_metrics` | Final test metrics, or `None` without test |
| `result.feature_selection`, `result.tuning` | Last split's optional search results |

Selection and tuning scores measure the data used to choose settings. The
optional test scores describe independent held-out performance. Evaluator
metrics are logged separately and are not part of these result summaries.

## Configure the workflow

Pass a [model factory](models.md), [metric](metrics.md) and prepared splitter.
`TrainingConfig` controls tuning, shared reporting and packaging. The
[task recipes](../configuration.md) show complete combinations.

| Need | Guide |
| --- | --- |
| Holdout, time-series CV or custom windows | [Data splitting](splitting.md) |
| Select columns or provide a ranker | [Feature selection](feature-selection.md) |
| Supply an Optuna search space or a standalone objective | [Parameter tuning](tuning.md) |
| Add reports or custom metrics/artifacts | [Evaluation](../evaluation/index.md) |
| Implement a training adapter | [Models](models.md) |
| Save, reload or inspect MLflow artifacts | [Tracking](tracking.md) |

## API reference

### Workflow

::: ml_toolkit.training.train_model

### Results

::: ml_toolkit.training.TrainingResult
    options:
      members: []

::: ml_toolkit.training.ValidationResult
    options:
      members: [mean_metrics, last_split]

::: ml_toolkit.training.SplitResult
    options:
      members: []
