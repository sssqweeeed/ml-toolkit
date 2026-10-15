# Explainability and tracking

Explainability helps inspect a fitted model; tracking preserves outputs in an
MLflow experiment. These helpers can also be used outside `train_model`.

## Explain a fitted model

Continue with `model` and `data` from the [model example](models.md#fit-and-predict-directly).
Pass the fitted native backend and a Polars feature frame to the explanation
helper.

```python
from ml_toolkit.explainability import shap_contributions

names, features, contributions = shap_contributions(model.native_model, data.X)
print(names)                 # ['age', 'region']
print(contributions.shape)   # (24, 3)
feature_effects = contributions[:, :-1]
base_values = contributions[:, -1]
```

There is one contribution per feature and a final base-value column. Their row
sum reconstructs the native raw score. For this RMSE regressor, that is the
prediction; for log-link or classification models, the raw scale differs from
the final transformed prediction.

Native contributions do not require the optional `shap` package. Install the
`shap` extra and use `calculate_shap_values` when you want a `shap.Explanation`
for the plotting helpers or `top_shap_features`.

## Save a table in MLflow

```python
import mlflow
import polars as pl

from ml_toolkit.tracking import log_polars_table

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("ml-toolkit-examples")
table = pl.DataFrame({"segment": ["north", "south"], "mae": [0.2, 0.3]})
with mlflow.start_run(run_name="segment-metrics") as run:
    log_polars_table(table, "tables/segment-metrics.json")
    print(run.info.run_id)
```

The call creates a local MLflow database if needed and writes a JSON artifact
containing the table's rows under the active run. The printed run ID identifies
the run containing `tables/segment-metrics.json`. Model adapters expose their
own model-packaging methods in the [model reference](models.md).

## API reference

::: ml_toolkit.explainability
    options:
      members: [shap_contributions, calculate_shap_values, top_shap_features]

::: ml_toolkit.tracking.log_polars_table
