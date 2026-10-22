# Training

`train_model` runs validation, optional feature selection and parameter tuning,
then fits a main model and evaluates optional test data. A
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
validation. This `Holdout` uses all non-test rows for its test-stage fit. Fractions
round upward. The workflow logs metrics and configuration to MLflow.

## Understand the run

1. Request `splitter.get_test()` once, then consume each validation pair in
   the splitter's iterator order.
2. Select columns and tune parameters when those extensions are supplied.
3. Fit the validation model and run enabled [evaluation reports](../evaluation/index.md).
4. Fit a fresh main model on `get_test().train`, including former validation
   and gap rows, and evaluate its optional `heldout` as test. This stage is
   `test` when heldout exists, otherwise `final_fit`.
5. With `fit_on_test=True`, request `splitter.get_final()` after test evaluation
   and fit a separate model on that full pool, including test.

The last yielded validation split supplies the final features, parameters,
iteration budget and classification threshold. The splitter determines final
training rows; they need not match the last validation train. The workflow
obtains the main training/test pair before fitting; your splitter owns sample
separation. Requesting `fit_on_test=True` without test raises `ValueError` before
any fit or MLflow run. [Reporting stage flags](configuration.md)
control additional reports; primary metrics always run.

## Use the result

| Result | Meaning |
| --- | --- |
| `result.model` | Main adapter fitted on all non-test rows, or all rows when test is absent |
| `result.final_model` | Additional model including test when `fit_on_test=True`, otherwise `None` |
| `result.params` | Last validation split's refit settings, shared by both models |
| `result.features` | Selected columns in fitted order |
| `result.validation.splits` | Validation results in execution order |
| `result.validation.mean_metrics` | Mean validation metrics across splits |
| `result.test_metrics` | Independent test metrics from the main model, or `None` without test |
| `result.feature_selection`, `result.tuning` | Last split's optional search results |

Selection and tuning scores measure the data used to choose settings. The
optional test scores describe independent held-out performance. Evaluator
metrics are logged separately and are not part of these result summaries.

## Optionally train including test

`TrainingConfig(fit_on_test=True)` adds a fresh fit on all eligible data after
independent test evaluation. Both models reuse the last validation split's
features, iteration budget, threshold and parameters, without another search.
Learned baselines and inflation state are fitted anew for each model.

| Test sample | fit_on_test | Stages after validation | model | final_model |
| --- | --- | --- | --- | --- |
| Present | False | test | All non-test rows | None |
| Present | True | test, then final_fit | All non-test rows | All eligible rows |
| Absent | False | final_fit | All eligible rows | None |
| Absent | True | ValueError before fitting or MLflow runs | — | — |

For example, reuse the holdout setup above:

<!-- example: training-including-test -->

```python
from dataclasses import replace

full_result = train_model(
    splitter=splitter,
    params=params,
    model_factory=CatBoostLogLinkRegressionModel,
    metric=metrics_frequency,
    run_name="frequency-including-test",
    config=replace(config, fit_on_test=True),
)
assert full_result.final_model is not None
```

`test_metrics` always belongs to the main model and is never replaced with
in-sample scores from the additional model. `final_fit` logs train metrics and
applicable train-only custom reports; held-out comparisons are skipped.
`ReportingConfig.on_test` and `on_final_fit` independently enable additional
reports; primary metrics are always calculated.

`TrackingConfig.log_test_model` controls test-stage packaging, and
`log_final_model` controls final_fit packaging, including the no-test case.
Both default to True. `registered_model_name` applies to the additional model
when requested, otherwise the main model, only if that model's packaging flag
is enabled. Disabling its packaging never registers the other model instead.

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
