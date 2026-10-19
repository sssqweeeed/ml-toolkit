# Build a model adapter

Implement `Model` when your project needs its own fitting or prediction policy.
The example below trains a CatBoost RMSE model in thousands of target units and
returns predictions in the original units. It composes a new adapter around a
built-in model and preserves that transformation when logging to MLflow.

The complete runnable files are `examples/custom_model.py` and
`examples/custom_model_serving.py` in the repository checkout. They use the
example [`ProjectPool`](api/data.md#prepare-a-pool); your own data class only
needs to satisfy the public `Pool` protocol.

## Run the complete example

From the installed repository checkout:

```bash
python -m examples.custom_model
```

Or run it from Python:

<!-- example: custom-model-demo -->

```python
from examples.custom_model import demo

demo()
```

<!-- output: custom-model-demo -->
```text
prediction shape: (40,)
validation MAE: 163.646
test MAE: 175.198
serving predictions match: True
```
<!-- /output -->

The example creates deterministic data, validates on a chronological holdout,
refits, logs the complete adapter, reloads it, and checks prediction equality.
MLflow uses a temporary local database and artifact directory that are removed
after the check. No remote tracking service or registry is required.

## Implement the training contract

Explicit inheritance from `Model` is unnecessary. A factory accepts a
`ModelParams` dictionary and returns a fresh object with these operations:

| Operation | What the example does | Contract to preserve |
| --- | --- | --- |
| `fit(train, validation)` | Divides both targets by the fixed scale; delegates fitting | Preserve row metadata, use validation only for selection, return the fitted adapter |
| `predict_pool(data)` | Multiplies native predictions back by the scale | Return one finite prediction per row in the metric's target units |
| `native_model` | Exposes the fitted `CatBoostRegressor` | Supply the native backend used by supported diagnostics |
| `best_iteration` | Delegates the validation-selected iteration | Return a zero-based index or `None` |
| `feature_importances(data)` | Delegates native feature ranking | Return one finite value for every current feature name |
| `get_refit_params()` | Reuses the selected iteration budget and disables early stopping | Return copied constructor settings, without learned state |
| `log(sample, config=...)` | Saves native weights, scale settings, and serving source | Preserve the adapter's complete prediction behavior |

`fit` uses `Pool.with_target`, so timestamps, weights, IDs, and categorical
metadata stay aligned. The scale is fixed project configuration; it is never
estimated from the validation or test rows. The factory supplies that same scale
to every fresh adapter. Selection and tuning therefore compare predictions in
the original target units.

??? note "Complete adapter and factory source"

    ```python
    --8<-- "examples/custom_model.py:adapter"
    ```

The source uses the built-in regressor for CatBoost's categorical preparation,
early stopping, feature importance, and refit policy. A custom implementation
can perform those operations itself, provided it keeps the same contracts.
Create new nested models and learned preprocessing objects in each factory
call; do not close over a fitted estimator or reuse mutable learned state.

## Connect the factory to training

The `demo()` function shows the complete call, including data preparation:

??? note "Executable workflow source"

    ```python
    --8<-- "examples/custom_model.py:workflow"
    ```

Pass `model_factory` to `train_model` together with your ordinary metric and
`TrainingConfig`. The metric receives predictions after the inverse scale
transformation. The same factory is used for feature selection, tuning trials,
validation, and the final fresh fit. `get_refit_params` transfers the selected
iteration budget to that last fit; the factory supplies the target scale again.

## Preserve predictions in serving

Logging only the inner CatBoost model would return values in scaled units.
The example's `log` writes the native model plus a concrete JSON settings file,
then uses MLflow Models From Code with this separate serving entrypoint:

??? note "Complete serving entrypoint"

    ```python
    --8<-- "examples/custom_model_serving.py"
    ```

The entrypoint restores feature order and categorical conversion, then applies
the inverse scale transformation. It depends only on pinned CatBoost, pandas,
NumPy, and MLflow packages. An explicit `TrackingConfig.conda_env` can supply your
own serving environment. The guide's executable check compares the reloaded
model against the adapter, so a missing transformation fails immediately.

## Backend and diagnostic constraints

The current `Model.native_model` annotation supports CatBoost and LightGBM
Boosters. A generic scikit-learn estimator does not satisfy that backend contract;
this guide does not promise that it can be passed directly to `train_model`.
`LightGBMWrapper` packages an already trained model and does not implement the
complete training adapter interface.

Shared diagnostics such as observed/fitted use `predict_pool` and see original
units. Native SHAP diagnostics and `ShapRanker` inspect `native_model` and see
scaled raw outputs in this example. Multiplication by a fixed positive scale
preserves relative feature rankings, but raw contributions must be multiplied
by the scale before interpreting them as original-unit contributions.

Before adopting your own adapter, check metadata alignment, finite one-value
predictions, fresh factory instances, selected iteration reuse, and serving
prediction equality. The public [`Model` reference](api/models.md#model-contract)
lists the full method signatures.
