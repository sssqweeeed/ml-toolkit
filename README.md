# ml-toolkit

One installable package for the shared parts of the current modeling projects:
model training, time-series validation, feature selection, Optuna tuning,
metrics, PSI, diagnostic plots, inflation strategies, and MLflow logging.
The import name is `ml_toolkit`.

## Installation

From this repository, install the package with `python -m pip install -e .`.
SHAP and LightGBM are optional: install `.[shap]`, `.[lightgbm]`, or `.[all]`
when those integrations are needed. CatBoost, Polars, Matplotlib, Optuna,
scikit-learn, and MLflow are installed by default.

## CodeGraph for Codex

This repository includes a project-local Codex MCP configuration for
[CodeGraph](https://github.com/colbymchenry/codegraph). Install the CodeGraph
CLI using its upstream instructions, then run `codegraph init` in the repository
root to build the local code index. Restart Codex to load the MCP server.
The generated `.codegraph/` index is ignored by Git.

## Training from a project

Projects own their data preparation, model settings, experiment names, and
MLflow tracking URI. Use the shared frozen `Pool` to pass prepared Polars data
to the toolkit. Existing custom containers remain supported through the
`TrainingData` protocol in `ml_toolkit.training.contracts`.

```python
from ml_toolkit.data import Pool  # Also available as: from ml_toolkit import Pool

data = Pool(
    X=prepared.select(["age", "region"]),
    y=prepared["target"],
    time=prepared["date"],  # Polars Date or Datetime
    weight=prepared["weight"],
    business_type=prepared["business_type"],  # Optional; defaults to None.
    cat_features=("region",),
    numerical_features=("age",),
)
```

`Pool` checks that rows are nonempty and aligned, timestamps have a Date or
Datetime dtype, and declared feature names exist without duplicates or overlap.
Feature lists may cover only a subset of columns. Training separately checks
ascending, non-null timestamps, finite targets, and valid weights.
`take`, `select_features`, and `with_target` return new pools and preserve aligned
metadata. A frozen dataclass prevents field reassignment; its contained Polars
objects are not deeply immutable.

`to_model_input(n_rows=None)` returns all rows as pandas data; a limit returns the
first rows and is capped at the pool size. Both this conversion and `to_catboost`
cast categorical columns to strings and fill null categories with `"unknown"`
without changing the source frame. `to_catboost` accepts an optional scalar or
per-row baseline and includes millisecond timestamps and row weights.

```python
import mlflow

from ml_toolkit.models import CatBoostRegressionModel
from ml_toolkit.metrics import metrics_frequency
from ml_toolkit.training import DiagnosticsOptions, train_model
from my_project.data import prepare_frequency_data

mlflow.set_experiment("Frequency")
data = prepare_frequency_data(raw_data)  # Your project prepares and returns a Pool.
result = train_model(
    data=data,
    params={"iterations": 200, "loss_function": "Poisson", "verbose": False},
    model_factory=CatBoostRegressionModel,
    metric=metrics_frequency,
    run_name="frequency",
    n_splits=4,
    select_features=None,
    diagnostics=DiagnosticsOptions(task="frequency", shap_top_n=None),
)
print(result.cross_validation.mean_metrics)
```

`train_model` starts an MLflow run, performs time-series cross-validation,
optionally selects features and tunes parameters, fits a final model, logs it,
and returns the fitted model and metrics. Each fold logs metrics and, when
requested, PSI tables and figures. `DiagnosticsOptions(psi=False, plots=False)`
disables those diagnostic artifacts without changing training.
When feature selection is enabled, it uses only the training rows of the first
evaluated fold. The selected features are then reused for all folds, tuning
trials, and the final model.

The project data object needs `X` (Polars DataFrame), `y`, `time`, and
`weight` (Polars Series), `cat_features`, `__len__`, `take(indices)`, and
`select_features(names)`. Bundled CatBoost models additionally need
`to_catboost(...)` and `to_model_input(...)`; inflation-adjusted fitting needs
`with_target(...)`. `business_type` is optional for grouped monthly plots.
The shared `Pool` implements all these operations. A custom model factory can
use a different project data representation if it satisfies the training
operations it invokes.

For classification, supply `CatBoostClassificationModel`,
`metrics_classification`, and `DiagnosticsOptions(task="classification")`.
The same `train_model` function performs the workflow. Optuna tuning is enabled
with `tune_params=True`, `param_space`, `metric_to_optimize`, and
`optimize_direction`.

For positive–unlabeled learning, label `1` means confirmed positive and `0`
means unlabeled. Use a PU loss, a prevalence estimate, `PUMetric`, and PU
diagnostics together:

```python
from ml_toolkit.metrics import PUMetric
from ml_toolkit.training import DiagnosticsOptions, train_model
from ml_toolkit.models import CatBoostClassificationModel

prior = 0.08  # Estimate from the target population, not the labeled fraction.
result = train_model(
    data=pu_data, model_factory=CatBoostClassificationModel,
    params={"loss": "nnPU", "pu_prior": prior, "iterations": 200,
            "depth": 5, "learning_rate": 0.05, "task_type": "CPU"},
    metric=PUMetric(prior, loss="nnPU"),
    diagnostics=DiagnosticsOptions(task="pu_classification"),
    run_name="pu", n_splits=3,
)
```

The PU metric logs positive risk, estimated negative risk, total PU risk,
known-positive capture at 1%, 5%, and 10% review capacity, and the actual
unlabeled share flagged at each cutoff. Score ties may make that share exceed
the target capacity. The diagnostics log score distributions and a capture
curve; they do not produce conventional precision, F1, gains, or calibration
plots. Known-positive capture approximates recall only if labeled positives
represent all positives. Confirmed precision and calibration require an
independently labeled audit sample. PU models do not select a decision threshold
from F1; pass `decision_threshold` in the model parameters if hard labels are
needed, and choose it from an audited set or an operational review capacity.

## Logging

The toolkit uses standard Python logging under the `ml_toolkit` namespace.
Applications configure handlers, formatting, and levels; importing the package
adds only a `NullHandler` and does not configure root logging. Package and module
loggers inherit levels and propagate to application handlers.

```python
import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("ml_toolkit").setLevel(logging.INFO)

# Enable details throughout the toolkit, or override a single module.
# logging.getLogger("ml_toolkit").setLevel(logging.DEBUG)
logging.getLogger("ml_toolkit.psi.analysis").setLevel(logging.DEBUG)
```

`INFO` reports workflow progress: feature selection, tuning, cross-validation
folds, and final training. `DEBUG` adds data sizes, feature names, metric
summaries, trial and nnPU iteration details, and model, PSI, SHAP, inflation,
plot, and artifact operations. `WARNING` suppresses routine toolkit progress
and details. Exceptions propagate normally without duplicate error logs.

These settings control toolkit messages. CatBoost, Optuna, and MLflow retain
their own output configuration. CatBoost feature selection still uses its
existing `logging_level="Verbose"`; nnPU's internal one-tree fits still use
`"Silent"`. Toolkit levels do not override those choices or caller-supplied
dependency settings. Python logging does not change MLflow metric or artifact
recording.

## PSI and plots

All plots are under `ml_toolkit.plots`, including PSI plots. PSI calculation is
under `ml_toolkit.psi`. These are the existing `ml-psi` algorithms, relocated
without a dependency on project data or a CatBoost Pool.

```python
from pathlib import Path
from ml_toolkit.psi import PSIAnalysis, PSIConfig, PSIFeature
from ml_toolkit.plots.psi import feature_psi_plot

config = PSIConfig((PSIFeature("age", "numerical"),), n_bins=10)
result = PSIAnalysis(config).fit(reference, time="date").compare(
    current, time="date"
)
feature_psi_plot(result.psi, Path("figures/psi.png"))
```

Other plot families are `classification`, `regression`, `ranking`,
`monitoring`, and `explainability`. The training diagnostics use those same
functions, so manual and automatic figures have one implementation.

## Package map

| Path | Purpose |
| --- | --- |
| `data.py` | Shared Pool container, aligned row operations, and model conversions. |
| `training/train.py` | Select, tune, validate, fit, and log one experiment. |
| `training/cross_validation.py` | Time-based folds, evaluation, and nested MLflow runs. |
| `training/feature_selection.py` | Model-driven feature elimination. |
| `training/tuning.py` | Optuna search through the same validation path. |
| `training/evaluation.py` | Metrics, PSI, plot generation, and MLflow artifacts. |
| `training/contracts.py` | TrainingData protocol for Pool and custom containers. |
| `training/results.py` | Result dataclasses. |
| `models/` | CatBoost classification/regression, PU learning, LightGBM serving, and inflation-adjusted models. |
| `metrics/` | Classification, frequency, severity, and ranking calculations. |
| `psi/` | Fitted levels, PSI scoring, distributions, and results. |
| `plots/` | Every chart, including PSI charts. |
| `inflation.py` | Constant-rate and manually indexed inflation strategies. |
| `explainability.py` | SHAP extraction from supported native models. |
| `tracking/` | MLflow table and model artifact helpers. |
| `tests/` | Behavioral and workflow integration tests. |

## Project boundaries

Project-specific input loading, joins, target definitions, Hydra runners,
inflation index values, and registered model names remain in their projects.
The toolkit accepts their prepared data and externally chosen settings.
