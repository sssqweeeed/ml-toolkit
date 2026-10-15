# Models

Adapters give training code a common interface for fitting, prediction, feature
importance, and MLflow packaging. You can fit an adapter directly, or pass its
class as `model_factory` to `train_model`, which creates fresh instances.

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

`predict_frame` accepts Polars or pandas features; `predict_pool` takes a pool.
Both require the fitted feature columns. Extra columns are ignored and fitted
feature order is restored. `model.native_model` exposes the fitted backend for
operations such as [SHAP contributions](tracking.md#explain-a-fitted-model).

For binary classification, `predict_pool` returns positive-class probabilities;
use `predict_labels_pool` for labels at the adapter's decision threshold.

## API reference

Import the CatBoost adapters, `InflationAdjustedModel`, and `Model` from
`ml_toolkit.models`.

::: ml_toolkit.models

### Optional LightGBM integration

Install the `lightgbm` extra and import `LightGBMWrapper` from
`ml_toolkit.models.lightgbm`.

::: ml_toolkit.models.lightgbm.LightGBMWrapper
