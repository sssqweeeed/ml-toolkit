# Build an evaluator

Implement `PredictionEvaluator` when a report needs project-specific metrics or
files. The two hooks have empty defaults: implement either or both.
This example records held-out weighted MAE and the prediction rows behind it.

## Implement the contract

The canonical implementation is importable from the source checkout:

```python
--8<-- "examples/custom_evaluator.py"
```

`evaluate()` returns finite scalar values keyed by sample and metric.
`write_artifacts()` receives the same aligned inputs plus an output directory
and shared plotting settings. Here it writes one CSV and returns its relative
path. The evaluator skips train-only calls and leaves the inputs unchanged.
It retains no rows or predictions on the instance; the workflow reuses that
instance for subsequent fits.

## Use it in training

Prepare the [real example data](index.md#example-data), then run:

<!-- example: custom-evaluator-workflow -->

```python
from pathlib import Path

import mlflow
import numpy as np
import polars as pl

from examples.custom_evaluator import PredictionAudit
from examples.gallery_data import load_bike_pool
from ml_toolkit import train_model
from ml_toolkit.config import ReportingConfig, TrackingConfig, TrainingConfig
from examples.custom_metric import weighted_mae
from ml_toolkit.models import CatBoostRegressionModel
from ml_toolkit.training import Holdout

mlflow.set_tracking_uri(f"sqlite:///{Path('mlflow.db').resolve()}")
mlflow.set_experiment("custom-evaluator")
data = load_bike_pool(download_if_missing=False)
result = train_model(
    splitter=Holdout(data, test_size="2012-01-01"),
    params={
        "iterations": 40,
        "depth": 4,
        "random_seed": 42,
        "thread_count": 2,
        "verbose": False,
        "allow_writing_files": False,
    },
    model_factory=CatBoostRegressionModel,
    metric=weighted_mae,
    run_name="prediction-audit",
    config=TrainingConfig(
        reporting=ReportingConfig(on_validation=False),
        tracking=TrackingConfig(log_test_model=False, log_final_model=False),
    ),
    prediction_evaluators=PredictionAudit(name="audit"),
)
client = mlflow.MlflowClient()
experiment = mlflow.get_experiment_by_name("custom-evaluator")
assert experiment is not None
runs = client.search_runs([experiment.experiment_id])
final_run = next(
    run for run in runs if run.data.tags["mlflow.runName"] == "test"
)
print("Additional metric:", round(final_run.data.metrics["audit/test/mae"], 3))
files = client.list_artifacts(final_run.info.run_id, "audit/test")
print("Artifact:", files[0].path)
local = client.download_artifacts(
    final_run.info.run_id, "audit/test/predictions.csv"
)
table = pl.read_csv(local, try_parse_dates=True)
print(
    table.head(4).with_columns(pl.col("predicted").round(2)).write_csv(), end=""
)
assert len(table) == int((data.time.dt.year() == 2012).sum())
np.testing.assert_allclose(
    np.average(
        abs(table["observed"] - table["predicted"]), weights=table["weight"]
    ),
    final_run.data.metrics["audit/test/mae"],
)
```

<!-- output: custom-evaluator-workflow -->
```text
Additional metric: 92.809
Artifact: audit/test/predictions.csv
timestamp,observed,predicted,weight
2012-01-01T00:00:00.000000,48.0,55.46,1.0
2012-01-01T01:00:00.000000,93.0,36.53,1.0
2012-01-01T02:00:00.000000,75.0,10.85,1.0
2012-01-01T03:00:00.000000,52.0,11.54,1.0
```
<!-- /output -->

The core `metric=` drives training results; `audit/test/mae` is the additional
logged metric. The CSV contains the actual test timestamps, targets, predictions,
and weights, so its rows reproduce the extra score.

Use a different name for each instance. Override `parameters()` only when your
report has settings to record; return explicit JSON-compatible values rather
than internal data or fitted state. This evaluator needs no override.
See the [shared contract](index.md#reporting-inputs-and-outputs) for names,
formats, scheduling, temporary paths, and error behavior.

## API reference

::: ml_toolkit.evaluation.PredictionEvaluator
    options:
      heading_level: 3
      members: [evaluate, write_artifacts, parameters]
