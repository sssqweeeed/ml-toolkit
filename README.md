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
MLflow tracking URI. `ml-toolkit` does not define a dataset or a `Pool` class.
The existing project `Pool` works because it provides the operations listed
below. Another project can provide an object with the same small interface.

```python
import mlflow

from ml_toolkit.models import CatBoostRegressionModel
from ml_toolkit.metrics import metrics_frequency
from ml_toolkit.training import DiagnosticsOptions, train_model
from my_project.data import prepare_frequency_data

mlflow.set_experiment("Frequency")
data = prepare_frequency_data(raw_data)  # Your project owns this function.
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

The project data object needs `X` (Polars DataFrame), `y`, `time`, and
`weight` (Polars Series), `cat_features`, `__len__`, `take(indices)`, and
`select_features(names)`. Bundled CatBoost models additionally need
`to_catboost(...)` and `to_model_input(...)`; inflation-adjusted fitting needs
`with_target(...)`. `business_type` is optional for grouped monthly plots.
These methods belong to the project, not to the toolkit. A custom model factory
can use a different project data representation if it satisfies the training
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
| `training/train.py` | Select, tune, validate, fit, and log one experiment. |
| `training/cross_validation.py` | Time-based folds, evaluation, and nested MLflow runs. |
| `training/feature_selection.py` | Model-driven feature elimination. |
| `training/tuning.py` | Optuna search through the same validation path. |
| `training/evaluation.py` | Metrics, PSI, plot generation, and MLflow artifacts. |
| `training/contracts.py` | Minimal project-owned data interface; no toolkit dataset class. |
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
