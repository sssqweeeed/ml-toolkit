# Inflation

Inflation strategies convert monetary values between price dates. Use a
constant annual rate for a simple assumption or a monthly index when you have
historical or planned index levels.

## Convert to a later price date

<!-- example: inflation-constant -->

```python
from datetime import date

import numpy as np
import polars as pl

from ml_toolkit.inflation import ConstantRateInflation

from_dates = pl.Series([date(2025, 1, 1)])
to_dates = pl.Series([date(2026, 1, 1)])
inflation = ConstantRateInflation(annual_rate=0.08)
adjusted = inflation.adjust(np.array([100.0]), from_dates, to_dates)
print(adjusted.round(2).tolist())  # [107.99]
```

<!-- output: inflation-constant -->
```text
[107.99]
```
<!-- /output -->

`0.08` means 8% annual compound growth. The result is slightly below `108`
because elapsed years use 365.25 days. Each row can have its own amount and date
pair; reversing the dates converts back to an earlier price level.

## Use a monthly index

<!-- example: inflation-monthly -->

```python
from ml_toolkit.inflation import ManualInflation

monthly = ManualInflation(
    {date(2025, 1, 1): 1.0, date(2025, 3, 1): 1.21}
)
adjusted = monthly.adjust(
    np.array([100.0]),
    pl.Series([date(2025, 1, 15)]),
    pl.Series([date(2025, 2, 28)]),
)
print(adjusted.round(2).tolist())  # [110.0]
```

<!-- output: inflation-monthly -->
```text
[110.0]
```
<!-- /output -->

Continuing with the imports above, February's interpolated index is `1.1`.
Every date within a month uses that month's level. Missing months between
anchors are interpolated in log space; requested months must stay within the
anchor range.

## Fit an inflation-adjusted regressor

Use `data` from the [data example](data.md#prepare-a-pool), treating its target
as a monetary amount observed at `data.time`. Fit the underlying regressor at
one reference price date:

<!-- example: inflation-model -->

```python
from ml_toolkit.config import HoldoutConfig
from ml_toolkit.models import CatBoostRegressionModel, InflationAdjustedModel
from ml_toolkit.training import split_holdout

split = split_holdout(data, HoldoutConfig(val_size=0.25))
adjusted_model = InflationAdjustedModel(
    CatBoostRegressionModel(
        {
            "iterations": 20, "depth": 2, "loss_function": "RMSE",
            "random_seed": 42, "thread_count": 2, "verbose": False,
            "allow_writing_files": False,
        }
    ),
    inflation=ConstantRateInflation(annual_rate=0.08),
    reference_date=date(2025, 1, 1),
).fit(split.train, split.val)
predictions = adjusted_model.predict_pool(split.val)
print(predictions.shape)  # (6,)

future = pl.DataFrame(
    {"age": [25], "region": ["north"], "prediction_date": [date(2026, 1, 1)]}
)
print(adjusted_model.predict_frame(future).shape)  # (1,)
```

<!-- output: inflation-model -->
```text
(6,)
(1,)
```
<!-- /output -->

Pool predictions use each row's observation date. Frame predictions require
the `prediction_date` column, whose name can be configured on the adapter.
With `ManualInflation`, all training, reference, and prediction months must
fall within the supplied index range.

## Log and reload for serving

Continuing with `adjusted_model`, `split`, and `future` above, configure a local
MLflow store and save both artifacts. `log()` returns the native model first and
the inflation-adjusted serving model second. Load the second artifact to retain
date-dependent predictions:

<!-- example: inflation-serving -->

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("ml-toolkit-inflation")
with mlflow.start_run():
    native_info, adjusted_info = adjusted_model.log(split.train)

loaded = mlflow.pyfunc.load_model(adjusted_info.model_uri)
np.testing.assert_allclose(
    loaded.predict(future.to_pandas()),
    adjusted_model.predict_frame(future),
)
```

This writes a local tracking database and model artifacts. The assertion checks
that the reloaded model preserves the predictions for the supplied price dates.

## API reference

Import strategies from `ml_toolkit.inflation`.

### Adjustment interface

::: ml_toolkit.inflation.InflationAdjuster
    options:
      filters: ['!^_', '!^to_config$']

### Inflation strategies

::: ml_toolkit.inflation.ConstantRateInflation
    options:
      filters: ['!^_', '!^to_config$']

::: ml_toolkit.inflation.ManualInflation
    options:
      filters: ['!^_', '!^to_config$']
