# Getting started

## Install

Python 3.11 or later is required. Clone the repository and install the package:

```bash
git clone https://github.com/sssqweeeed/ml-toolkit.git
cd ml-toolkit
python -m pip install -e .
```

The distribution is named `ml-toolkit`; the Python import is `ml_toolkit`.
CatBoost, Polars, NumPy, pandas, Matplotlib, scikit-learn, Optuna, Numba, and
MLflow are installed by default.

Optional integrations are installed from the same checkout:

| Extra | Command | Purpose |
| --- | --- | --- |
| SHAP | `python -m pip install -e '.[shap]'` | Diagnostic SHAP explanations |
| LightGBM | `python -m pip install -e '.[lightgbm]'` | LightGBM integration |
| Both | `python -m pip install -e '.[all]'` | SHAP and LightGBM |
| CUDA | `python -m pip install -e '.[cuda]'` | Single-GPU CatBoost uPU training |

CUDA training requires a compatible NVIDIA GPU and driver. CPU training does
not require CUDA, and the `all` extra does not include it.

## Train a model

Predict hourly rental counts from the original [UCI Bike Sharing dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset).
Run from the installed checkout above: the reusable loader lives in `examples/`,
not in the installed package. Explicitly download the source once:

```bash
python -m examples.prepare_data --dataset bike
```

The loader caches `hour.csv` under `.cache/uci-bike-sharing`. It keeps all 17,379
recorded hours from 2011–2012 with their original timestamps and rental counts.
Weather and calendar fields are predictors; `casual` and `registered` are excluded
because their sum reveals the target. Each hour has weight one.

July–December 2012 is held out for testing. The last 20% of earlier observations
forms validation; the main model refits on all pre-test hours. No test observations
are used to choose training iterations.

<!-- example: getting-started-train -->

```python
from ml_toolkit.training import Holdout
from datetime import datetime

import numpy as np
import mlflow

from examples.gallery_data import load_bike_pool
from ml_toolkit import train_model
from ml_toolkit.config import TrackingConfig, TrainingConfig
from ml_toolkit.metrics import metrics_frequency
from ml_toolkit.models import CatBoostLogLinkRegressionModel

# Run the explicit preparation command above once before reading the cache.
data = load_bike_pool(download_if_missing=False)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("ml-toolkit-quickstart")

result = train_model(
    splitter=Holdout(data, val_size=0.2, test_size="2012-07-01"),
    params={
        "iterations": 100,
        "depth": 5,
        "loss_function": "Poisson",
        "random_seed": 42,
        "thread_count": 2,
        "verbose": False,
        "allow_writing_files": False,
    },
    model_factory=CatBoostLogLinkRegressionModel,
    metric=metrics_frequency,
    run_name="hourly-rentals",
    config=TrainingConfig(tracking=TrackingConfig(log_test_model=False, log_final_model=False)),
)
print(
    f"Validation deviance: {result.validation.mean_metrics['mean_poisson_deviance']:.2f}"
)
assert result.test_metrics is not None
print(f"Test deviance: {result.test_metrics['mean_poisson_deviance']:.2f}")
test = data.take(np.flatnonzero((data.time >= datetime(2012, 7, 1)).to_numpy()))
predictions = result.model.predict_pool(test)
print("Observed rentals:", test.y.head(5).to_list())
print("Predicted rentals:", predictions[:5].round(1).tolist())
```

<!-- output: getting-started-train -->
```text
Validation deviance: 70.59
Test deviance: 72.23
Observed rentals: [149.0, 93.0, 90.0, 33.0, 4.0]
Predicted rentals: [48.7, 48.0, 47.8, 47.2, 47.5]
```
<!-- /output -->

Poisson deviance summarizes count prediction error; lower is better. The first
five predictions are genuine future-period observations, in rentals per hour.
Validation and test cover different seasons, so their scores need not match.
This small model demonstrates the workflow; it is not a tuned demand forecast.

`Pool` is included in the installed package. Pass your own prepared data to its
constructor; see the [data guide](api/data.md) for fields and operations.
The example creates a local `mlflow.db` and logs metrics and configuration.
Model packaging is disabled here; `TrackingConfig(log_test_model=True)` also
saves the main model and its inference dependencies.

## Reproduce the other examples

From the checkout, install optional integrations and the example-only Excel
reader, then prepare the remaining sources explicitly:

```bash
python -m pip install -e '.[all]' -r requirements-examples.txt
python -m examples.prepare_data
```

Bike Sharing supports count and demand-classification workflows; the gallery
also uses bundled handwritten digits and original Cora citation records.
[UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online%2Bretail)
(CC BY 4.0; Daqing Chen, 2015) provides recorded invoice timestamps and item
prices in sterling for [monetary and inflation examples](api/inflation.md).
Its source workbook is approximately 23 MB. `openpyxl` is required only to read
that workbook; it is not a package runtime dependency.

`python -m examples.render_examples --check` executes the recipes and checks
captured results and figures. Dataset caches stay local; recipe execution does
not implicitly download missing sources. Remove `--check` to regenerate output.

## Prepare your own data

- Keep features, targets, timestamps, and weights aligned and nonempty.
- Sort observations by ascending time. Use Polars `Date` or `Datetime` timestamps.
- Supply finite targets and finite, nonnegative weights with positive total.
- List categorical and numerical features explicitly when needed. Keep IDs and
  reporting groups outside `X` unless they are intentional model inputs.
- Configure your own MLflow URI and experiment before training.

`Pool.take`, `select_features`, and `with_target` return new pools with aligned
metadata. See the [data API](api/data.md) for the complete contract.

## Choose the next step

- [Task recipes](configuration.md) choose a model, metric and reports for your target.
- [Training](api/training.md) explains the result and execution order.
- [Data splitting](api/splitting.md) supplies validation and final-fit pools.
- [Evaluation and plots](evaluation/index.md) shows what each report produces.
- [Models](api/models.md) covers prediction, packaging and custom adapters.
- [Configuration](api/configuration.md) lists shared settings and defaults.
