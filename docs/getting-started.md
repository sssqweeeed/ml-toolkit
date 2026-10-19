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

This example creates synthetic frequency data, reserves the latest 20% for
testing, and uses one chronological holdout for validation.

<!-- example: getting-started-train -->

```python
from datetime import date, timedelta

import mlflow
import numpy as np
import polars as pl

from examples.project_pool import ProjectPool
from ml_toolkit import train_model
from ml_toolkit.config import HoldoutConfig, TrackingConfig, TrainingConfig
from ml_toolkit.metrics import metrics_frequency
from ml_toolkit.models import CatBoostLogLinkRegressionModel

rng = np.random.default_rng(42)
n_rows = 240
age = rng.integers(18, 80, size=n_rows)
region = rng.choice(["north", "south"], size=n_rows)
rate = np.exp(-0.5 + 0.015 * (age - 40) + 0.2 * (region == "north"))
frame = pl.DataFrame(
    {
        "age": age,
        "region": region,
        "target": rng.poisson(rate),
        "date": [date(2025, 1, 1) + timedelta(days=i) for i in range(n_rows)],
        "weight": np.ones(n_rows),
    }
)
data = ProjectPool(
    X=frame.select("age", "region"),
    y=frame["target"],
    time=frame["date"],
    weight=frame["weight"],
    cat_features=("region",),
    numerical_features=("age",),
)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("ml-toolkit-quickstart")

result = train_model(
    data=data,
    params={
        "iterations": 40,
        "depth": 4,
        "loss_function": "Poisson",
        "random_seed": 42,
        "thread_count": 2,
        "verbose": False,
        "allow_writing_files": False,
    },
    model_factory=CatBoostLogLinkRegressionModel,
    metric=metrics_frequency,
    run_name="frequency-holdout",
    config=TrainingConfig(
        validation=HoldoutConfig(val_size=0.2),
        test_size=0.2,
        tracking=TrackingConfig(log_final_model=False),
    ),
)

print(f"Validation deviance: {result.validation.mean_metrics['mean_poisson_deviance']:.4f}")
assert result.test_metrics is not None
print(f"Test deviance: {result.test_metrics['mean_poisson_deviance']:.4f}")
predictions = result.model.predict_pool(data)
print(predictions[:5].round(4).tolist())
```

<!-- output: getting-started-train -->
```text
Validation deviance: 1.2006
Test deviance: 1.3690
[0.6797, 0.7921, 0.7012, 0.6933, 0.6933]
```
<!-- /output -->

`ProjectPool` is the runnable data implementation in this repository's
`examples/project_pool.py`. It is outside the installed package. Copy it into
your project or implement the [Pool interface](api/data.md) on your existing
container; inheritance is unnecessary.

The example creates a local `mlflow.db` and logs metrics and configuration.
Model packaging is disabled to keep the example small; set
`TrackingConfig(log_final_model=True)` to also log the final model and its
inference dependencies. Predictions at the end include training rows and are
an inference example; use `result.test_metrics` for held-out evaluation.

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

Use [configuration by task](configuration.md) for validation splits, feature
selection, tuning, reports, and model packaging. The
[training reference](api/training.md) describes the returned results and extension
interfaces; the [configuration reference](api/configuration.md) lists defaults
and constraints.
