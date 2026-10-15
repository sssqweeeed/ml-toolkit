# Models

Import the CatBoost adapters, `InflationAdjustedModel`, and the `Model` protocol
from `ml_toolkit.models`. The training workflow calls the adapter's explicit
training, prediction, ranking, and logging operations.

::: ml_toolkit.models

## Optional LightGBM integration

Install the `lightgbm` extra and import `LightGBMWrapper` from
`ml_toolkit.models.lightgbm`.

::: ml_toolkit.models.lightgbm.LightGBMWrapper
