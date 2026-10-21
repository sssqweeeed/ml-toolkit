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
print("Later price level:", adjusted.round(2).tolist())
restored = inflation.adjust(adjusted, to_dates, from_dates)
print("Original price level:", restored.round(2).tolist())
np.testing.assert_allclose(restored, [100.0])
```

<!-- output: inflation-constant -->
```text
Later price level: [107.99]
Original price level: [100.0]
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
print(adjusted.round(2).tolist())
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

## Prepare recorded retail prices

Run from the installed repository checkout and prepare the source explicitly:

```bash
python -m pip install -r requirements-examples.txt
python -m examples.prepare_data --dataset retail
```

The reusable loader reads [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online%2Bretail)
(Daqing Chen, 2015; CC BY 4.0). `UnitPrice` is sterling per item and `InvoiceDate`
is the recorded transaction time. The approximately 23 MB workbook and normalized
records stay in `.cache/uci-online-retail`; reading Excel requires `openpyxl`, an
example dependency rather than a package runtime dependency.

We exclude cancelled invoices and lines with nonpositive quantity or price.
A reproducible approximately 10% invoice sample retains all eligible lines of each
selected invoice: the first eight bytes of its ID's SHA-256 digest, interpreted
big-endian, must have remainder zero modulo ten. Original values and timestamps
are retained. December 2011 is excluded because it is incomplete.
Training ends before September 2011, validation covers September, and testing
covers October–November. Invoice groups never cross those partitions.

<!-- example: inflation-data -->

```python
from examples.retail_data import load_retail_splits

retail = load_retail_splits(download_if_missing=False)
print("Train / validation / test lines:",
      len(retail.train), len(retail.validation), len(retail.test))
print("First held-out prices (GBP/item):", retail.test.y.head(5).to_list())
```

<!-- output: inflation-data -->
```text
Train / validation / test lines: 29847 4282 15210
First held-out prices (GBP/item): [0.85, 7.08, 4.95, 4.25, 4.95]
```
<!-- /output -->

The three predictors are product code, country, and purchased quantity. Invoice
IDs, customer IDs, timestamps, and target-derived amounts are excluded from model
features. Each invoice line has weight one, so metrics describe line-level prices.
This is a price-prediction demonstration, not an insurance-claims dataset.

## Fit an inflation-adjusted regressor

Continue with `retail` above and the NumPy, Polars, date, and inflation imports
from the first calculation. Fit at a common reference price date, then predict
held-out transaction prices at their recorded dates:

<!-- example: inflation-model -->

```python
from ml_toolkit.models import CatBoostRegressionModel, InflationAdjustedModel

adjusted_model = InflationAdjustedModel(
    CatBoostRegressionModel(
        {
            "iterations": 60, "depth": 4, "loss_function": "RMSE",
            "random_seed": 42, "thread_count": 2, "verbose": False,
            "allow_writing_files": False,
        }
    ),
    inflation=ConstantRateInflation(annual_rate=0.08),
    reference_date=date(2011, 9, 1),
).fit(retail.train, retail.validation)
predictions = adjusted_model.predict_pool(retail.test)
print("Predicted prices (GBP/item):", predictions[:5].round(2).tolist())
print("Held-out MAE (GBP/item):", round(float(
    np.mean(np.abs(predictions - retail.test.y.to_numpy()))
), 2))

# Hold product, country, and quantity fixed; vary only the requested price date.
requested_dates = pl.Series("prediction_date", [date(2011, 9, 1), date(2012, 9, 1)])
future = retail.test.X[[0, 0]].with_columns(requested_dates)
date_predictions = adjusted_model.predict_frame(future)
assert date_predictions[1] > date_predictions[0] > 0
print("Same item at requested price dates:", date_predictions.round(2).tolist())
conversion = adjusted_model.inflation.factor(requested_dates.head(1), requested_dates.tail(1))
np.testing.assert_allclose(date_predictions[1:], date_predictions[:1] * conversion)
```

<!-- output: inflation-model -->
```text
Predicted prices (GBP/item): [1.42, 5.08, 4.87, 7.22, 9.98]
Held-out MAE (GBP/item): 2.18
Same item at requested price dates: [1.41, 1.52]
```
<!-- /output -->

The 8% rate is an **assumed conversion policy**, not an estimate of historical UK
inflation. The date comparison demonstrates the adapter's behavior; it does not
establish that the assumption improves predictive accuracy. The two requested
serving dates do not replace any recorded training or evaluation timestamps.

Pool predictions use each row's observation date. Frame predictions require
`prediction_date`, whose name can be configured on the adapter. With
`ManualInflation`, all training, reference, and requested months must lie within
the supplied index range.

## Log and reload for serving

Continue with the fitted `adjusted_model`, `retail`, and `future` above. `log()`
returns the native model first and the adjusted serving model second. Load the
second artifact to retain date-dependent predictions:

<!-- example: inflation-serving -->

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("ml-toolkit-inflation")
with mlflow.start_run():
    native_info, adjusted_info = adjusted_model.log(retail.train)

loaded = mlflow.pyfunc.load_model(adjusted_info.model_uri)
np.testing.assert_allclose(
    loaded.predict(future.to_pandas()),
    adjusted_model.predict_frame(future),
)
```

This writes a local tracking database and model artifacts. The assertion checks
that the reloaded model preserves both requested price-date predictions.

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
