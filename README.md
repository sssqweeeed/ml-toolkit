# ml-toolkit

A Python toolkit for model training, time-series validation, feature selection,
Optuna tuning, metrics, diagnostic plots, PSI, inflation adjustment, and MLflow
integration.

[Documentation](https://sssqweeeed.github.io/ml-toolkit/) · [Getting started](https://sssqweeeed.github.io/ml-toolkit/getting-started/)

## Installation

Requires Python 3.11 or later.

```bash
git clone https://github.com/sssqweeeed/ml-toolkit.git
cd ml-toolkit
python -m pip install -e .
```

The Python import is `ml_toolkit`.

## Quickstart

Train a frequency model on synthetic data with chronological validation and a
future test split:

<!-- example: readme-quickstart -->

```python
from datetime import date

import mlflow
import polars as pl

from ml_toolkit.data import Pool
from ml_toolkit import train_model
from ml_toolkit.config import HoldoutConfig, TrackingConfig, TrainingConfig
from ml_toolkit.metrics import metrics_frequency
from ml_toolkit.models import CatBoostLogLinkRegressionModel

frame = pl.DataFrame({
    "feature": [i % 10 for i in range(40)],
    "target": [float(i % 3) for i in range(40)],
    "date": pl.date_range(date(2025, 1, 1), date(2025, 2, 9), eager=True),
})
data = Pool(
    X=frame.select("feature"),
    y=frame["target"],
    time=frame["date"],
    weight=pl.Series([1.0] * len(frame)),
    numerical_features=("feature",),
)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("ml-toolkit-quickstart")

result = train_model(
    data=data,
    params={
        "iterations": 20, "depth": 2, "loss_function": "Poisson", "random_seed": 42,
        "thread_count": 2, "verbose": False, "allow_writing_files": False,
    },
    model_factory=CatBoostLogLinkRegressionModel,
    metric=metrics_frequency,
    run_name="frequency",
    config=TrainingConfig(
        validation=HoldoutConfig(val_size=0.2),
        test_size=0.2,
        tracking=TrackingConfig(log_final_model=False),
    ),
)
print(f"Validation deviance: {result.validation.mean_metrics['mean_poisson_deviance']:.4f}")
assert result.test_metrics is not None
print(f"Test deviance: {result.test_metrics['mean_poisson_deviance']:.4f}")
```

<!-- output: readme-quickstart -->
```text
Validation deviance: 0.7952
Test deviance: 1.0407
```
<!-- /output -->

`Pool` is the package's ready-to-use data container; see the data guide in the
documentation for its fields and operations.

The example logs metrics and configuration to a local MLflow store using
`mlflow.db`. See the [documentation](https://sssqweeeed.github.io/ml-toolkit/)
for model packaging, tuning, reports, and other tasks.
