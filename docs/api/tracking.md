# Explainability and tracking

Explainability helps inspect a fitted model; tracking preserves outputs in an
MLflow experiment. These helpers can also be used outside `train_model`.

## Explain a fitted model

Continue with `model` and `data` from the [model example](models.md#fit-and-predict-directly).
Pass the fitted native backend and a Polars feature frame to the explanation
helper.

<!-- example: tracking-native -->

```python
from ml_toolkit.explainability import shap_contributions

names, features, contributions = shap_contributions(model.native_model, data.X)
print(names)                 # ['age', 'region']
print(contributions.shape)   # (24, 3)
feature_effects = contributions[:, :-1]
base_values = contributions[:, -1]
```

<!-- output: tracking-native -->
```text
['age', 'region']
(24, 3)
```
<!-- /output -->

There is one contribution per feature and a final base-value column. Their row
sum reconstructs the native raw score. For this RMSE regressor, that is the
prediction; for log-link or classification models, the raw scale differs from
the final transformed prediction.

Native contributions do not require the optional `shap` package. Install the
`shap` extra and use `calculate_shap_values` when you want a `shap.Explanation`
for the plotting helpers or `top_shap_features`:

<!-- example: tracking-shap -->

```python
from ml_toolkit.config import ShapConfig
from ml_toolkit.explainability import calculate_shap_values, top_shap_features

explanation = calculate_shap_values(
    model.native_model, data.X, config=ShapConfig(max_samples=24, seed=0)
)
important_features = top_shap_features(explanation, top_n=2)
print(len(important_features))  # 2
```

<!-- output: tracking-shap -->
```text
2
```
<!-- /output -->

Calculation samples at most `max_samples` rows using `seed`; defaults are 10,000
rows and seed `0`. The explanation retains sampled feature values and native
raw-score contributions. See [plot SHAP explanations](plots.md#plot-shap-explanations)
for a complete example of importance, beeswarm, and dependence plots.

## Save a table in MLflow

<!-- example: tracking-table -->

```python
import mlflow
import polars as pl

from ml_toolkit.tracking import log_polars_table

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("ml-toolkit-examples")
table = pl.DataFrame({"segment": ["north", "south"], "mae": [0.2, 0.3]})
with mlflow.start_run(run_name="segment-metrics") as run:
    log_polars_table(table, "tables/segment-metrics.json")
    saved = mlflow.artifacts.load_dict(
        f"runs:/{run.info.run_id}/tables/segment-metrics.json"
    )
    print(saved)
```

<!-- output: tracking-table -->
```text
{'rows': [{'segment': 'north', 'mae': 0.2}, {'segment': 'south', 'mae': 0.3}]}
```
<!-- /output -->

The call creates a local MLflow database if needed and writes a JSON artifact
containing the table's rows under the active run. The example reads back the
saved payload to verify its contents. Model adapters expose their
own model-packaging methods in the [model reference](models.md).

## API reference

### Native feature contributions

::: ml_toolkit.explainability.shap_contributions

### SHAP explanations and selection

::: ml_toolkit.explainability.calculate_shap_values

::: ml_toolkit.explainability.top_shap_features

### Experiment tables

::: ml_toolkit.tracking.log_polars_table
