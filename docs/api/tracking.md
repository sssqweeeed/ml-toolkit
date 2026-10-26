# Tracking

`train_model` records its experiment in MLflow. Set the tracking URI and
experiment before calling it; the toolkit uses the active run as a parent when
one already exists. The same table helpers also work outside training.

## Inspect a training run

A workflow has a parent named by `run_name`, a `validation` run containing
`validation/split_<index>` runs, and a `test` run when test exists.
Without test, the main fit is `final_fit`; with test and `fit_on_test=True`,
a separate `final_fit` follows test evaluation. Selection history is
stored in `selection/history.json` inside each split; tuning uses nested
`tuning` and `trial_<number>` runs. The final fit uses the
last yielded split's settings and the final data supplied by the splitter.

| Where to look | Contents |
| --- | --- |
| Parent `configuration/requested.json` | Shared training configuration |
| Parent `configuration/splitter.json` | Splitter class and explicit parameters |
| Parent `configuration/evaluators.json` | Prediction evaluator classes, names and parameters |
| Parent `configuration/data_analyzers.json` | Data analyzer classes, names and parameters |
| Parent `configuration/resolved.json` | Final selected features and model settings |
| `validation` | Averages of primary validation metrics |
| `validation/split_<index>` | Train/validation metrics, optional selection, trials and reports |
| `test` | Main model train/test metrics, reports and optional model |
| `final_fit` | All-data train metrics, applicable reports and optional model; no independent test metrics |

Primary metrics drive training and tuning; evaluator and analyzer metrics are additional
named measurements. See [Evaluation](../evaluation/index.md) and [Data analysis](../data-analysis/index.md) for
report scheduling, metric names and artifact paths. The
[complete task recipe](../configuration.md#complete-frequency-example) shows
selection, tuning, SHAP and final model packaging together.

`TrackingConfig` controls packaging for trials, validation, test and final
fitting. `log_test_model=True` saves the test-stage main model;
`log_final_model=True` saves the final-fit model, including when test is absent.
Registration selects the additional model when trained, otherwise the main
model, and requires that model's logging flag. It never falls back to the other
model. Each stage logs its own row counts and timestamp ranges. Model methods preserve their prediction transformation when saving
and loading; use the [model serving guide](models.md#log-and-reload-for-serving)
or the [custom adapter guide](custom-model.md) for that contract.

## Prepare a fitted model

This independent example uses recorded Bike Sharing hours. The last 20% are
held out; the preceding data contains a trailing 25% validation window. Run
`python -m examples.prepare_data --dataset bike` from the checkout once.

<!-- example: tracking-setup -->

```python
import numpy as np
from examples.gallery_data import load_bike_pool
from ml_toolkit.models import CatBoostRegressionModel
from ml_toolkit.training import Holdout

data = load_bike_pool(download_if_missing=False)
splitter = Holdout(data, val_size=0.25, test_size=0.2)
part = next(splitter.get_validation())
test = splitter.get_test().heldout
assert test is not None
model = CatBoostRegressionModel(
    {
        "iterations": 40,
        "depth": 4,
        "loss_function": "RMSE",
        "random_seed": 42,
        "thread_count": 2,
        "verbose": False,
        "allow_writing_files": False,
    }
).fit(part.train, part.heldout)
```

## Save a table in MLflow

<!-- example: tracking-table -->

```python
import mlflow
import polars as pl

from ml_toolkit.tracking import log_polars_table

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("ml-toolkit-examples")
predictions = model.predict_pool(test)
table = (
    test.X.select("working_day")
    .with_columns(
        pl.Series("absolute_error", np.abs(test.y.to_numpy() - predictions))
    )
    .group_by("working_day")
    .agg(
        pl.col("absolute_error").mean().round(3).alias("mae"),
        pl.len().alias("hours"),
    )
    .sort("working_day")
)
with mlflow.start_run(run_name="segment-metrics") as run:
    log_polars_table(table, "tables/segment-metrics.json")
    saved = mlflow.artifacts.load_dict(
        f"runs:/{run.info.run_id}/tables/segment-metrics.json"
    )
    print(saved)
```

<!-- output: tracking-table -->
```text
{'rows': [{'working_day': 0, 'mae': 89.048, 'hours': 1127}, {'working_day': 1, 'mae': 101.836, 'hours': 2349}]}
```
<!-- /output -->

The table summarizes held-out absolute error in rentals per hour by the
recorded working-day flag, alongside the number of hours in each group.
The call creates a local MLflow database if needed and writes a JSON artifact
containing the table's rows under the active run. The example reads back the
saved payload to verify its contents. Model adapters expose their
own model-packaging methods in the [model reference](models.md).

## API reference

::: ml_toolkit.tracking.load_psi_baseline

::: ml_toolkit.tracking.log_polars_table

## Load a model's PSI baseline

With `ReportingConfig(psi=True)`, each saved model includes a `psi_baseline`
artifact containing fitted feature levels and unweighted reference counts.
The existing `TrackingConfig.log_trial_models`, `log_validation_models`,
`log_test_model`, and `log_final_model` switches control this packaging; no
additional tracking switch is needed. `ReportingConfig.on_*` controls reports,
so disabling a stage's reports does not remove the saved model's baseline.
`features=()` explicitly disables both levels and their baseline.

Use the same immutable model URI for serving and monitoring:

```python
import mlflow
from ml_toolkit.tracking import load_psi_baseline

model_uri = "models:/risk_model/17"  # Or ModelInfo.model_uri returned by log().
model = mlflow.pyfunc.load_model(model_uri)
baseline = load_psi_baseline(model_uri)

# current_X is a Polars frame in the original training feature representation.
current_levels = baseline.transformer.transform(current_X)
result = baseline.compare(current_X)
```

The loader also accepts a local model package. It reads the named artifact from
`MLmodel` without loading native model weights or executing serving code. Install
a compatible `ml-toolkit` in the monitoring environment. Use a fixed model
version instead of independently resolving a mutable registry alias for serving
and monitoring.

Each baseline uses all training rows after feature selection, independently of
`input_example_rows`. Validation/test rows are excluded unless they are part of
the subsequent final fit's training population. A train-only final fit still
packages its baseline. Inflation-adjusted models include the same reference in
both their native and wrapper packages.

The package contains `manifest.json` (format version, scoring settings and
reference metadata), `levels.json` (typed fitted transformation rules), and
`reference_counts.parquet` (feature/level counts, including zero counts).
It contains no source rows and requires no separate PyFunc model. Graphs and
evaluation distributions remain run artifacts. Missing baselines raise
`FileNotFoundError`; malformed or unsupported state raises `ValueError`.
Existing models remain loadable for predictions without migration.
