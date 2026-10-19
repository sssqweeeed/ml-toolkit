# Models

CatBoost adapters share fitting, prediction, feature importance, and MLflow
packaging operations. Fit an adapter directly, or pass its class as
`model_factory` to `train_model`, which creates fresh instances.
The optional LightGBM wrapper packages an already fitted native regressor.

## Choose an adapter

| Adapter | Use it for |
| --- | --- |
| `CatBoostRegressionModel` | General regression with CatBoost's native loss and initialization |
| `CatBoostLogLinkRegressionModel` | Poisson or Tweedie regression with a fitted log-mean baseline |
| `CatBoostClassificationModel` | Binary classification with probability and label predictions |
| `CatBoostPUClassificationModel` | Positive-unlabeled learning with an explicit population prior |
| `InflationAdjustedModel` | A regressor composed with an inflation strategy |

## Fit and predict directly

Use `data` from the [data example](data.md#prepare-a-pool). This fits a general
RMSE regressor and predicts two new rows without targets or timestamps.

<!-- example: models-regression -->

```python
import polars as pl

from ml_toolkit.config import HoldoutConfig
from ml_toolkit.models import CatBoostRegressionModel
from ml_toolkit.training import split_holdout

split = split_holdout(data, HoldoutConfig(val_size=0.25))
model = CatBoostRegressionModel(
    {
        "iterations": 20,
        "depth": 2,
        "loss_function": "RMSE",
        "random_seed": 42,
        "thread_count": 2,
        "verbose": False,
        "allow_writing_files": False,
    }
)
model.fit(split.train, split.val)
future = pl.DataFrame({"age": [25, 35], "region": ["north", "south"]})
predictions = model.predict_frame(future)
print(predictions.shape)  # (2,)
```

<!-- output: models-regression -->
```text
(2,)
```
<!-- /output -->

`predict_frame` accepts Polars or pandas features; `predict_pool` takes a pool.
Both require the fitted feature columns. Extra columns are ignored and fitted
feature order is restored. `model.native_model` exposes the fitted backend for
operations such as [SHAP contributions](tracking.md#explain-a-fitted-model).

For binary classification, `predict_pool` returns positive-class probabilities;
use `predict_labels_pool` for labels at the adapter's decision threshold.

<!-- example: models-classification -->

```python
from ml_toolkit.models import CatBoostClassificationModel

binary = data.with_target((data.y > 0).cast(pl.Float64))
binary_split = split_holdout(binary, HoldoutConfig(val_size=0.25))
classifier = CatBoostClassificationModel(
    {
        "iterations": 20,
        "depth": 2,
        "loss_function": "Logloss",
        "random_seed": 42,
        "thread_count": 2,
        "verbose": False,
        "allow_writing_files": False,
    }
).fit(binary_split.train, binary_split.val)
probabilities = classifier.predict_frame(future)
labels = classifier.predict_labels(future)
print(probabilities.shape, labels.shape)  # (2,) (2,)
```

<!-- output: models-classification -->
```text
(2,) (2,)
```
<!-- /output -->

Without an explicit `decision_threshold`, validation chooses the classifier's
cutoff using weighted F1. Labels use `probability >= threshold`; ties in F1
choose the higher threshold. Zero-weight rows do not affect the choice. With
no positive-class validation weight, the cutoff is `0.5` and F1 is `0`.
`classifier.threshold` and `classifier.validation_f1` expose these fitted
values, which are preserved when the model is saved and reloaded. The latter
is `None` when threshold tuning is skipped: an explicit threshold, no
validation, or PU training.
Positive-unlabeled training uses `CatBoostPUClassificationModel` with an
explicit population prior; see its constructor constraints below. For monetary
targets across dates, follow the [inflation composition example](inflation.md).

## Log and reload for serving

Continue with the fitted regression `model`, `split`, and `future` above.
Configure MLflow before logging and keep the returned model URI to load later.

<!-- example: models-serving -->

```python
import mlflow
import numpy as np

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("ml-toolkit-model-examples")
with mlflow.start_run(run_name="direct-regression"):
    info = model.log(split.train)

served = mlflow.pyfunc.load_model(info.model_uri)
served_predictions = served.predict(future.to_pandas())
np.testing.assert_allclose(served_predictions, model.predict_frame(future))
```

Logging writes native model artifacts, a serving example, and inference
dependencies. The loaded model's `predict` accepts a pandas frame; call
`predict_frame` or `predict_pool` on local adapters. `TrackingConfig` controls
artifact naming, registration, and the example size.

## Optional LightGBM integration

Install the `lightgbm` extra. Fit LightGBM with its own training API, then wrap
the fitted estimator for logging and inference. `LightGBMWrapper` is not a
`train_model` factory. Continue with `data` from the data example:

<!-- example: models-lightgbm -->

```python
import lightgbm as lgb
import mlflow
import numpy as np

from ml_toolkit.models.lightgbm import LightGBMWrapper

native = lgb.LGBMRegressor(
    n_estimators=10,
    min_child_samples=2,
    random_state=42,
    n_jobs=2,
    verbosity=-1,
)
features = data.to_model_input()
features["region"] = features["region"].astype("category")
native.fit(features, data.y.to_numpy(), sample_weight=data.weight.to_numpy())
wrapper = LightGBMWrapper(
    native,
    numerical_features=data.numerical_features,
    cat_features=data.cat_features,
)
predictions = wrapper.predict(None, features)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("ml-toolkit-model-examples")
with mlflow.start_run(run_name="lightgbm-regression"):
    lightgbm_info = wrapper.log(data)

loaded = mlflow.pyfunc.load_model(lightgbm_info.model_uri)
serving_input = data.to_model_input()
np.testing.assert_allclose(loaded.predict(serving_input), predictions)
```

Numerical and categorical feature names must partition the native model's
features exactly. The wrapper restores feature order and converts configured
categorical columns to pandas categories before prediction.
Pass ordinary string columns to the loaded MLflow model; its saved input
signature validates them before the wrapper performs categorical conversion.

## API reference

Import the CatBoost adapters, `InflationAdjustedModel`, and `Model` from
`ml_toolkit.models`.

### Regression adapters

::: ml_toolkit.models.CatBoostRegressionModel
    options:
      members:
        - fit
        - predict_frame
        - predict_pool
        - predict
        - log
        - native_model
        - best_iteration
        - feature_names
        - cat_features
        - numerical_features
        - baseline
        - feature_importances
        - get_refit_params

::: ml_toolkit.models.CatBoostLogLinkRegressionModel
    options:
      inherited_members: true
      members:
        - fit
        - predict_frame
        - predict_pool
        - predict
        - log
        - baseline

### Classification adapters

::: ml_toolkit.models.CatBoostClassificationModel
    options:
      members:
        - fit
        - predict_frame
        - predict_pool
        - predict_labels
        - predict_labels_pool
        - predict
        - log
        - threshold
        - validation_f1
        - native_model
        - best_iteration
        - feature_names
        - cat_features
        - numerical_features
        - baseline
        - feature_importances
        - get_refit_params

::: ml_toolkit.models.CatBoostPUClassificationModel
    options:
      inherited_members: true
      members:
        - fit
        - predict_frame
        - predict_pool
        - predict_labels
        - predict_labels_pool
        - predict
        - log
        - threshold
        - pu_loss
        - pu_prior

### Inflation and serving

::: ml_toolkit.models.InflationAdjustedModel
    options:
      members:
        - fit
        - predict_frame
        - predict_pool
        - predict
        - log
        - native_model
        - best_iteration
        - feature_importances
        - get_refit_params

::: ml_toolkit.models.lightgbm.LightGBMWrapper
    options:
      members:
        - predict
        - log
        - native_model
        - feature_names

### Model contract

Follow [build a model adapter](../custom-model.md) for a complete implementation,
factory, and serving parity check. Custom factories accept a `ModelParams` dictionary and return a fresh adapter
implementing `Model`. Keep learned state in that adapter; the workflow creates
new instances for selection, tuning, validation, and final fitting. A factory
for `InflationAdjustedModel` must also create a fresh wrapped model and
inflation strategy using the composition in the [inflation guide](inflation.md).

::: ml_toolkit.models.Model
    options:
      members:
        - fit
        - predict_pool
        - native_model
        - best_iteration
        - feature_importances
        - get_refit_params
        - log
