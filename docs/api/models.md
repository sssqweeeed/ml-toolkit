# Models

CatBoost adapters share fitting, prediction, feature importance, and MLflow
packaging operations. Fit an adapter directly, or pass its class as
`model_factory` to `train_model`, which creates fresh instances.
The optional LightGBM wrapper packages an already fitted native regressor.

## Choose an adapter

| Adapter | Use it for |
| --- | --- |
| `CatBoostRegressionModel` | General regression with CatBoost's native loss and initialization |
| `CatBoostLogLinkRegressionModel` | Poisson, Tweedie or Gamma regression with a fitted log-mean baseline |
| `CatBoostClassificationModel` | Binary classification with probability and label predictions |
| `CatBoostPUClassificationModel` | Positive-unlabeled learning with an explicit population prior |
| `InflationAdjustedModel` | A regressor composed with an inflation strategy |

## Prepare real observations

Run these examples from an installed source checkout after
`python -m examples.prepare_data`. The regression and classification examples
use all 17,379 recorded hourly observations from the
[UCI Bike Sharing dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset).
Targets are rentals per recorded hour; weights are one per hour. The eleven
weather/calendar features exclude counts that reveal the target. Timestamps
come from the original date and hour columns.

<!-- example: models-data -->

```python
from ml_toolkit.training import Holdout
from examples.gallery_data import load_bike_pool

data = load_bike_pool(download_if_missing=False)
splitter = Holdout(data, val_size=0.25, test_size=0.2)
development = splitter.get_test()
split = next(splitter.get_validation())
test = development.heldout
```

The last 20% of hours are untouched test rows. The preceding observations are
split chronologically into training and validation. The following examples
show predictions on those test rows; validation chooses fitting settings.

## Gamma regression

Use `loss_function="Gamma"` (or `loss="Gamma"`) for a strictly positive target,
such as claim severity conditional on a claim. Both regression adapters accept
it. Prefer `CatBoostLogLinkRegressionModel` for its fitted initial log mean:

<!-- example: models-gamma -->

```python
import numpy as np

from examples.retail_data import load_retail_splits
from ml_toolkit.metrics import metrics_severity
from ml_toolkit.models import CatBoostLogLinkRegressionModel

retail = load_retail_splits(download_if_missing=False)
gamma = CatBoostLogLinkRegressionModel(
    {
        "loss_function": "Gamma",
        "task_type": "CPU",
        "iterations": 40,
        "learning_rate": 0.05,
        "depth": 3,
        "random_seed": 42,
        "thread_count": 2,
        "verbose": False,
        "allow_writing_files": False,
    }
)
gamma.fit(retail.train, retail.validation)
predicted_means = gamma.predict_pool(retail.test)
scores = metrics_severity(
    retail.test.y.to_numpy(),
    predicted_means,
    retail.test.weight.to_numpy(),
    power=2.0,
)
print("Observed unit prices (GBP):", retail.test.y.head(5).round(2).to_list())
print("Predicted unit prices (GBP):", predicted_means[:5].round(2).tolist())
print("Test Gamma deviance:", round(scores.mean_tweedie_deviance, 4))
assert np.isfinite(predicted_means).all() and (predicted_means > 0).all()
```

<!-- output: models-gamma -->
```text
Observed unit prices (GBP): [0.85, 7.08, 4.95, 4.25, 4.95]
Predicted unit prices (GBP): [1.33, 3.93, 4.58, 5.9, 5.9]
Test Gamma deviance: 0.5101
```
<!-- /output -->

The shared [retail preparation](inflation.md#prepare-recorded-retail-prices)
uses recorded sterling unit prices and invoice timestamps. It keeps a deterministic
approximately 10% sample of eligible invoice groups, excludes cancellations and nonpositive prices
or quantities, trains before September 2011, validates in September, and tests
in October–November. Stock code, country and quantity are inputs; price is
excluded.

Gamma minimizes `log(mu) + y / mu`. The toolkit creates its custom objective
only for training; constructor and refit parameters retain the string `"Gamma"`.
CPU derivatives and deviance are compiled with Numba. Defaults are Plain boosting,
Newton leaf estimation and one leaf estimation iteration. Explicit compatible
parameters override these defaults.

Targets must remain finite and strictly positive after CatBoost's float32
conversion, even for zero-weight rows. Weights must be finite, nonnegative and
have a positive total after conversion. Values are not clipped to satisfy these
requirements. Use Tweedie with `1 < variance_power < 2` when zero targets belong
in the problem.

The default `eval_metric` (also selected by `eval_metric="Gamma"`) is weighted
Gamma deviance. Early stopping and best-iteration selection use this metric.
An explicitly supplied alternative metric receives the raw log means, following
CatBoost's custom-objective convention. For toolkit validation/reporting, use
`functools.partial(metrics_severity, power=2.0)` from `ml_toolkit.metrics`.

Predictions default to positive means, with the log-link baseline applied once.
Set the adapter's `prediction_type="RawFormulaVal"` to request log means instead;
`prediction_type="Exponent"` explicitly requests means. A raw native CatBoost
model with this custom objective defaults to raw scores: when calling
`gamma.native_model.predict` directly, pass `prediction_type="Exponent"` for means.
The general `CatBoostRegressionModel` uses native initialization and rejects
`boost_from_average=True` for Gamma; use the log-link adapter for mean initialization.

For one CUDA GPU, use the following training parameters. This block only constructs
an adapter; fitting it requires the GPU setup described below:

<!-- example: models-gamma-gpu-config -->

```python
gpu_gamma = CatBoostLogLinkRegressionModel(
    {"loss_function": "Gamma", "task_type": "GPU", "devices": "0"}
)
```

GPU derivatives run in a CUDA kernel. Gamma deviance is evaluated on CPU and
requires transferring predictions from GPU. Install the `cuda` extra and use a
compatible CUDA toolkit and driver, as for uPU. Multiple devices and automatic
CPU fallback are not supported.

Native `.cbm` plus toolkit state, MLflow models, and `InflationAdjustedModel`
preserve the prediction mode. Bundled inference does not require Numba or CUDA.
Older version-1 model states remain loadable; refitting a restored Gamma model
recreates the training objective from its saved string parameters.

## Fit and predict directly

Continue with the Bike `split` and `test` prepared above. This fits an RMSE
regressor and predicts held-out hours from features alone. Observed counts are
shown beside the predictions in the same units.

<!-- example: models-regression -->

```python
import polars as pl

from ml_toolkit.models import CatBoostRegressionModel

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
).fit(split.train, split.heldout)
preview = test.take([0, 8, 16, 24, 32])
future = preview.X
predictions = model.predict_frame(future)
print("Observed rentals:", preview.y.to_list())
print("Predicted rentals:", predictions.round(1).tolist())
```

<!-- output: models-regression -->
```text
Observed rentals: [283.0, 448.0, 7.0, 280.0, 500.0]
Predicted rentals: [197.3, 273.0, 14.1, 200.0, 293.3]
```
<!-- /output -->

`predict_frame` accepts Polars or pandas features; `predict_pool` takes a pool.
Both require the fitted feature columns. Extra columns are ignored and fitted
feature order is restored. `model.native_model` exposes the fitted backend for
operations such as [SHAP contributions](../evaluation/explainability.md).

For binary classification, `predict_pool` returns positive-class probabilities;
use `predict_labels_pool` for labels at the adapter's decision threshold.

<!-- example: models-classification -->

```python
from ml_toolkit.models import CatBoostClassificationModel

# A high-demand hour means at least 100 recorded rentals.
binary_train = split.train.with_target((split.train.y >= 100).cast(pl.Float64))
binary_validation = split.heldout.with_target(
    (split.heldout.y >= 100).cast(pl.Float64)
)
classifier = CatBoostClassificationModel(
    {
        "iterations": 40,
        "depth": 4,
        "loss_function": "Logloss",
        "random_seed": 42,
        "thread_count": 2,
        "verbose": False,
        "allow_writing_files": False,
    }
).fit(binary_train, binary_validation)
probabilities = classifier.predict_frame(future)
labels = classifier.predict_labels(future)
print("Observed high demand:", (preview.y >= 100).cast(pl.Int64).to_list())
print("Probabilities:", probabilities.round(3).tolist())
print("Decision threshold:", round(classifier.threshold, 3))
print("Predicted labels:", labels.tolist())
assert (labels == (probabilities >= classifier.threshold)).all()
```

<!-- output: models-classification -->
```text
Observed high demand: [1, 1, 0, 1, 1]
Probabilities: [0.963, 0.969, 0.053, 0.962, 0.978]
Decision threshold: 0.171
Predicted labels: [1, 1, 0, 1, 1]
```
<!-- /output -->

Without an explicit `decision_threshold`, binary validation labels choose the
classifier's cutoff using weighted F1. `CrossEntropy` also accepts soft targets
in `[0, 1]`; with soft validation labels the cutoff stays at `0.5` and F1 tuning
is skipped. Explicit thresholds work with either kind of label.
Labels use `probability >= threshold`; ties in F1
choose the higher threshold. Zero-weight rows do not affect the choice. With
no positive-class validation weight, the cutoff is `0.5` and F1 is `0`.
`classifier.threshold` and `classifier.validation_f1` expose these fitted
values, which are preserved when the model is saved and reloaded. The latter
is `None` when threshold tuning is skipped: an explicit threshold, no
validation, soft validation labels, or PU training.
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
the fitted estimator for logging and inference. `LightGBMServingModel` is not a
`train_model` factory. Continue with the Bike training and test split above:

<!-- example: models-lightgbm -->

```python
import lightgbm as lgb
import mlflow
import numpy as np

from ml_toolkit.models.lightgbm import LightGBMServingModel

native = lgb.LGBMRegressor(
    n_estimators=10,
    min_child_samples=2,
    random_state=42,
    n_jobs=2,
    verbosity=-1,
)
features = split.train.to_model_input()
for column in split.train.cat_features:
    features[column] = features[column].astype("category")
native.fit(
    features,
    split.train.y.to_numpy(),
    sample_weight=split.train.weight.to_numpy(),
)
wrapper = LightGBMServingModel(
    native,
    numerical_features=split.train.numerical_features,
    cat_features=split.train.cat_features,
)
serving_input = test.to_model_input(categorical_as_strings=True)
predictions = wrapper.predict(None, serving_input)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("ml-toolkit-model-examples")
with mlflow.start_run(run_name="lightgbm-regression"):
    lightgbm_info = wrapper.log(split.train)

loaded = mlflow.pyfunc.load_model(lightgbm_info.model_uri)
np.testing.assert_allclose(loaded.predict(serving_input), predictions)
print("Held-out predictions preserved:", predictions[:5].round(1).tolist())
```

<!-- output: models-lightgbm -->
```text
Held-out predictions preserved: [164.2, 164.2, 164.2, 164.2, 245.6]
```
<!-- /output -->

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

::: ml_toolkit.models.lightgbm.LightGBMServingModel
    options:
      merge_init_into_class: true
      members:
        - predict
        - log
        - native_model
        - feature_names

### Custom adapters

[Build a model adapter](custom-model.md) covers the complete contract, a reusable
implementation and factory, and a saved/reloaded prediction check. Factories
create fresh adapters for selection, tuning, validation and final fitting.
For an inflation-adjusted adapter, each call also creates a fresh wrapped model
and strategy; see [custom inflation](inflation.md#implement-an-inflation-strategy).
