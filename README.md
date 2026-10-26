# ml-toolkit

A Python toolkit for model training, time-series validation, feature selection,
Optuna tuning, metrics, diagnostic plots, PSI, inflation adjustment, and MLflow
integration.

[Stable documentation](https://sssqweeeed.github.io/ml-toolkit/stable/) · [Development documentation](https://sssqweeeed.github.io/ml-toolkit/dev/) · [Getting started](https://sssqweeeed.github.io/ml-toolkit/stable/getting-started/)

## Installation

Requires Python 3.11 or later.

```bash
python -m pip install 'ml-toolkit @ git+https://github.com/sssqweeeed/ml-toolkit.git@v0.1.1'
```

The Python import is `ml_toolkit`. Installation from a release tag requires Git.
Wheels are also available from [GitHub Releases](https://github.com/sssqweeeed/ml-toolkit/releases).

## Quickstart

Check out the matching release for its example loaders, then cache the original
[UCI Bike Sharing data](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)
and predict hourly rental counts using recorded weather and calendar features:

```bash
git clone --branch v0.1.1 --depth 1 https://github.com/sssqweeeed/ml-toolkit.git
cd ml-toolkit
python -m examples.prepare_data --dataset bike
```

The loader excludes counts that reveal the target. Training and validation use
hours before July 2012; July–December 2012 is the future test period.

<!-- example: readme-quickstart -->

```python
from ml_toolkit.training import Holdout
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
```

<!-- output: readme-quickstart -->
```text
Validation deviance: 70.59
Test deviance: 72.23
```
<!-- /output -->

`Pool` holds aligned features, targets, weights and time; see the
[data guide](https://sssqweeeed.github.io/ml-toolkit/api/data/).
Choose validation and final datasets with a
[splitter](https://sssqweeeed.github.io/ml-toolkit/api/splitting/), then attach
[prediction evaluators](https://sssqweeeed.github.io/ml-toolkit/evaluation/) and
[data analyzers](https://sssqweeeed.github.io/ml-toolkit/data-analysis/) for plots,
additional metrics and tables. Both interfaces support project implementations.

The example logs metrics and configuration to a local MLflow store using
`mlflow.db`. See the [documentation](https://sssqweeeed.github.io/ml-toolkit/)
for model packaging, tuning, reports, and other tasks.
