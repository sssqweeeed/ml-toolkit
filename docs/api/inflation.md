# Inflation

Inflation strategies convert monetary values between price dates. Use a
constant annual rate for a supplied assumption, a monthly index for historical
or planned levels, or a learned common index estimated from project observations.

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

monthly = ManualInflation({date(2025, 1, 1): 1.0, date(2025, 3, 1): 1.21})
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

## Learn a common price index

`LogLinearInflation` and `SplineInflation` estimate a weighted Gamma mean with
a log link:

`log(E[amount | features, date]) = intercept + feature effects + trend(date)`

The first strategy has a linear time effect, giving a constant compound growth
rate inside training. The second uses a regularized cubic B-spline to represent
a changing growth rate. Both fit original positive amounts, rather than a
regression on log-transformed targets.

**The proportional price change is shared by all objects.** Features help
distinguish a change in product composition from price growth: an increasing
share of expensive products need not imply inflation. When comparing the same
object at two dates, its feature effect cancels, leaving the multiplier
`exp(trend(to_date) - trend(from_date))`. Features are required only for fitting;
`factor()` and `adjust()` still accept dates without features.

Numeric controls have standardized linear effects; categorical controls use
sparse one-hot encoding according to `Pool.cat_features`. Numeric missing values
use training medians; all-missing numeric columns are dropped. Interactions,
object-specific trends and separate seasonal components are not modeled.
Missing price-determining characteristics or a strong relationship between time
and the available characteristics can limit interpretation of the estimated trend.

The following reproducible data is synthetic, with Gamma price variation,
changing product composition and a known time effect; it is not measured
inflation. Continue with the NumPy, Polars and date imports above:

<!-- example: inflation-learned -->

```python
import json

from examples.inflation_benchmark import synthetic_pool
from ml_toolkit.inflation import (
    LogLinearInflation,
    SplineInflation,
    inflation_from_config,
)

training = synthetic_pool(months=36, rows_per_month=25)
linear = LogLinearInflation(control_features=["size", "category"]).fit(training)
smooth = SplineInflation(
    n_knots=5,
    alpha=0.001,
    extrapolation="damped",
    damping_half_life_years=1.0,
).fit(training)
original = pl.Series([date(2017, 1, 1)])
requested = pl.Series([date(2018, 1, 1)])
for strategy in (linear, smooth):
    changed = strategy.adjust(np.array([100.0]), original, requested)
    np.testing.assert_allclose(
        strategy.adjust(changed, requested, original), [100.0]
    )

saved = json.loads(json.dumps(smooth.to_config(), allow_nan=False))
restored = inflation_from_config(saved)
np.testing.assert_allclose(
    restored.factor(original, requested), smooth.factor(original, requested)
)
```

`fit(data)` takes a `Pool`. `control_features=None` uses every available column,
an explicit sequence chooses controls, and `control_features=[]` fits dates
only. Every preprocessing step uses positive-weight training rows only. Targets
must be finite and strictly positive; weights must be finite and nonnegative
with positive total. Zero-weight rows do not set the fitted time range or knots.
The linear method needs at least two distinct positive-weight timestamps; the
spline needs at least `n_knots` (default five). These target and weight restrictions
do not apply to the fixed strategies' no-op fit.

Both methods use elapsed time in 365.25-day years, including fractional days.
Timezone-aware timestamps refer to UTC instants. Defaults are `alpha=0.001`
(L2 regularization of time and control coefficients), `extrapolation="linear"`,
and `damping_half_life_years=1.0`; the spline degree is fixed at three.

### Choose an extrapolation policy

The forecast horizon is determined by the requested dates; there is no fixed
one-year limit. `extrapolation` selects behavior outside positive-weight training:

| Policy | Behavior |
| --- | --- |
| `linear` | Continue the boundary log-index tangent, preserving its growth rate |
| `constant` | Hold the boundary index level |
| `error` | Reject dates outside the fitted range |
| `damped` | After the last training date, exponentially decay the boundary log-index slope toward zero; before the first date, extrapolate linearly |

For `damped`, the future slope halves every `damping_half_life_years`. This
approaches a constant price level, not a long-run inflation target. The spline
polynomial is never continued beyond its training boundaries. Policies define
one index function, so conversions remain reversible and agree when performed
through an intermediate date. Unrepresentable factors raise `ValueError`.

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
print(
    "Train / validation / test lines:",
    len(retail.train),
    len(retail.validation),
    len(retail.test),
)
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
            "iterations": 60,
            "depth": 4,
            "loss_function": "RMSE",
            "random_seed": 42,
            "thread_count": 2,
            "verbose": False,
            "allow_writing_files": False,
        }
    ),
    inflation=ConstantRateInflation(annual_rate=0.08),
    reference_date=date(2011, 9, 1),
).fit(retail.train, retail.validation)
predictions = adjusted_model.predict_pool(retail.test)
print("Predicted prices (GBP/item):", predictions[:5].round(2).tolist())
print(
    "Held-out MAE (GBP/item):",
    round(float(np.mean(np.abs(predictions - retail.test.y.to_numpy()))), 2),
)

# Hold product, country, and quantity fixed; vary only the requested price date.
requested_dates = pl.Series(
    "prediction_date", [date(2011, 9, 1), date(2012, 9, 1)]
)
future = retail.test.X[[0, 0]].with_columns(requested_dates)
date_predictions = adjusted_model.predict_frame(future)
assert date_predictions[1] > date_predictions[0] > 0
print("Same item at requested price dates:", date_predictions.round(2).tolist())
conversion = adjusted_model.inflation.factor(
    requested_dates.head(1), requested_dates.tail(1)
)
np.testing.assert_allclose(
    date_predictions[1:], date_predictions[:1] * conversion
)
```

<!-- output: inflation-model -->
```text
Predicted prices (GBP/item): [1.42, 5.08, 4.87, 7.22, 9.98]
Held-out MAE (GBP/item): 2.18
Same item at requested price dates: [1.41, 1.52]
```
<!-- /output -->

The 8% annual rate is an assumed conversion policy, not a historical UK
inflation estimate.

Pool predictions use each row's observation date. Frame predictions require
`prediction_date`, whose name can be configured on the adapter. With
`ManualInflation`, all training, reference, and requested months must lie within
the supplied index range.

### Use a learned strategy in a model factory

Keep strategy settings in the factory, and create a fresh strategy on every
call. The workflow can then refit it independently for feature selection,
validation and final fitting. Continue with `retail`, `adjusted_model` and the
model imports above:

<!-- example: inflation-learned-model -->

```python
from ml_toolkit.types import ModelParams


def learned_model(params: ModelParams) -> InflationAdjustedModel:
    return InflationAdjustedModel(
        CatBoostRegressionModel(params),
        inflation=LogLinearInflation(alpha=0.001),
        reference_date=date(2011, 9, 1),
    )


learned_retail = learned_model(adjusted_model.model.params).fit(
    retail.train, retail.validation
)
assert np.isfinite(learned_retail.predict_pool(retail.test)).all()
```

`InflationAdjustedModel` passes only the original training pool to the strategy.
Validation amounts are normalized using that fitted index. With
`control_features=None`, feature selection also determines the controls available
to each strategy fit. Explicit controls must be present in each supplied pool.
Do not pass strategy settings as native CatBoost parameters.

### Compare strategies at calendar forecast horizons

Start with the linear strategy and compare the spline and extrapolation settings
on chronological validation. The benchmark holds CatBoost parameters fixed for
four methods: no adjustment, an assumed constant rate, a learned linear trend,
and a learned spline trend. Each origin fits on preceding observations only.

```bash
python -m examples.inflation_benchmark --output inflation-benchmark.csv
python -m examples.inflation_benchmark --extrapolation damped --damping-half-life-years 1
python -m examples.inflation_benchmark --dataset retail --min-train-months 6 --step-months 1 --horizons 1 3
```

The default synthetic benchmark tests 1, 3, 6 and 12 months. Horizon 12 evaluates
the twelfth month after the training cutoff, rather than accumulating twelve
months of observations. Incomplete horizons are skipped. The real-data command
requires the explicitly prepared Online Retail cache above and never downloads
implicitly. Its history is shorter than a year and cannot establish twelve-month
forecast quality.

Individual fold results include weighted MAE, RMSE, Gamma deviance and predicted
over observed total amounts. Synthetic results also measure error in the known
log-index, relative to the final training date. Printed summaries average fold
metrics equally across origins. The constant rate defaults to an assumed 8%,
not an estimate. There is no guarantee a learned adjustment improves the final
regressor, and synthetic index recovery is not evidence of real-world superiority.

For reference, running the synthetic and retail commands above with
`--iterations 60` and the default strategy settings produced these mean fold
MAEs. Synthetic monetary units are arbitrary; retail units are GBP per item.

| Strategy | Synthetic, 12 months | Retail, 1 month | Retail, 3 months |
| --- | ---: | ---: | ---: |
| No adjustment | 113.360 | 1.879 | 2.061 |
| Assumed 8% rate | 88.364 | 1.852 | 2.063 |
| Learned linear | 72.337 | 1.805 | 1.941 |
| Learned spline | 93.952 | 1.846 | 2.067 |

These settings illustrate a tradeoff: the retail linear adjustment reduced MAE
but its predicted/observed amount ratio moved further below one (from 0.900 to
0.850 at one month and from 0.888 to 0.783 at three months). The spline did not
outperform the linear strategy. Select settings using the metric appropriate to
the project rather than assuming any adjustment is uniformly better.

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
The same logging API supports the two learned strategies: their settings,
temporal coefficients, spline knots and fitted boundaries are saved as JSON.
Feature transformers and the Gamma estimator are not needed for serving, and
loading does not refit either estimator.

## Implement an inflation strategy

Subclass `InflationAdjuster` and implement `factor(from_dates, to_dates)`:
return one finite positive multiplier per aligned date pair. The inherited
`adjust()` multiplies amounts by these factors. Override `fit()` when learning
parameters from `fit(data: Pool)`, using only its training pool; keep learned
state in that strategy instance. A model factory must create a fresh strategy
for each fit, including final refitting.

This project implementation accepts a separate compound rate for each calendar
year. A full year applies exactly `1 + rate`, including leap years; a partial
year compounds by the fraction of that year's calendar days. Datetime inputs
use their local dates. Requested dates must lie within the supplied consecutive
years, including the following January 1 boundary.

??? note "Complete strategy source"

    ```python
    --8<-- "examples/custom_inflation.py"
    ```

These rates illustrate a supplied conversion policy, not measured inflation.
Continue with the NumPy, Polars and date imports from the first calculation:

<!-- example: inflation-custom-calculation -->

```python
from examples.custom_inflation import AnnualRateTable

annual = AnnualRateTable({2024: 0.03, 2025: 0.07})
original_dates = pl.Series([date(2024, 1, 1)])
later_dates = pl.Series([date(2026, 1, 1)])
converted = annual.adjust(np.array([100.0]), original_dates, later_dates)
print("After 3% then 7%:", converted.round(2).tolist())
restored = annual.adjust(converted, later_dates, original_dates)
print("Reversed conversion:", restored.round(2).tolist())
np.testing.assert_allclose(restored, [100.0])
```

<!-- output: inflation-custom-calculation -->
```text
After 3% then 7%: [110.21]
Reversed conversion: [100.0]
```
<!-- /output -->

### Fit with a project strategy

Continue with `retail`, `mlflow` and the model imports above. This example uses
the same recorded retail observations and chronological boundaries as the
built-in strategy. Rates are fixed inputs; inherited `fit()` validates dates
without learning from prices. The normal workflow creates separate strategy
instances for validation and final fitting.

<!-- example: inflation-custom-training -->

```python
from examples.custom_metric import weighted_mae
from examples.retail_data import load_retail_pool
from ml_toolkit import train_model
from ml_toolkit.config import TrackingConfig, TrainingConfig
from ml_toolkit.training import Holdout
from ml_toolkit.types import ModelParams


def annual_model(params: ModelParams) -> InflationAdjustedModel:
    return InflationAdjustedModel(
        CatBoostRegressionModel(params),
        inflation=AnnualRateTable({2010: 0.02, 2011: 0.08, 2012: 0.05}),
        reference_date=date(2011, 9, 1),
    )


retail_data = load_retail_pool(download_if_missing=False)
annual_result = train_model(
    splitter=Holdout(
        retail_data,
        val_size=len(retail.validation),
        test_size="2011-10-01",
    ),
    params={
        "iterations": 30,
        "depth": 4,
        "loss_function": "RMSE",
        "random_seed": 42,
        "thread_count": 2,
        "verbose": False,
        "allow_writing_files": False,
    },
    model_factory=annual_model,
    metric=weighted_mae,
    run_name="retail-annual-policy",
    config=TrainingConfig(tracking=TrackingConfig(log_test_model=False, log_final_model=False)),
)
annual_predictions = annual_result.model.predict_pool(retail.test)
assert annual_result.test_metrics is not None
print(
    "Custom-policy test MAE (GBP/item):",
    round(annual_result.test_metrics["mae"], 2),
)
print("Predicted prices (GBP/item):", annual_predictions[:5].round(2).tolist())
```

<!-- output: inflation-custom-training -->
```text
Custom-policy test MAE (GBP/item): 2.08
Predicted prices (GBP/item): [1.09, 3.9, 5.45, 7.21, 7.74]
```
<!-- /output -->

The final model includes September's validation rows and predicts prices at
October–November transaction dates. Metrics and configuration are logged to the
local MLflow store configured above.

Bundled model packaging reconstructs `ConstantRateInflation`, `ManualInflation`,
`LogLinearInflation`, and `SplineInflation`. A custom strategy such as `AnnualRateTable` works for
fitting and prediction, but `InflationAdjustedModel.log()` raises `ValueError`
for it. Keep trial, validation and final model logging disabled, as above, or
provide your own serving implementation that saves and restores the strategy.
See [the model adapter guide](custom-model.md#preserve-predictions-in-serving)
for an executable persistence pattern.

## API reference

Import strategies from `ml_toolkit.inflation`.

### Adjustment interface

::: ml_toolkit.inflation.InflationAdjuster
    options:
      members: [fit, factor, adjust]

### Inflation strategies

::: ml_toolkit.inflation.ConstantRateInflation
    options:
      members: [fit, factor, adjust]

::: ml_toolkit.inflation.ManualInflation
    options:
      members: [fit, factor, adjust]

::: ml_toolkit.inflation.LogLinearInflation
    options:
      members: [fit, factor, adjust, to_config]

::: ml_toolkit.inflation.SplineInflation
    options:
      members: [fit, factor, adjust, to_config]

### Migrating custom strategies

`InflationAdjuster.fit(dates, y, sample_weight)` has been replaced by
`fit(data: Pool)`. Update overrides to read `data.time`, `data.y`, `data.weight`,
and optionally `data.X`. Fixed strategies can continue inheriting the no-op fit,
which now validates `data.time`. There is no `fit_pool` compatibility method;
the date-pair APIs and existing fixed-strategy artifact formats are unchanged.
