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
    object_id=prepared["customer_id"],  # Optional; required for post-fit evaluation.
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

### Custom post-fit evaluation

Use the separate `post_fit_evaluator` argument for business metrics that need
object IDs or externally prepared reference data. Define an object with an
`evaluate` method; its constructor, internal data, and helper methods are
project-owned. Explicit inheritance from `PostFitEvaluator` is unnecessary.
The contracts live in `ml_toolkit.training.contracts` and are also exported by
`ml_toolkit.training`.

```python
import numpy as np
import polars as pl

from ml_toolkit.metrics import metrics_classification
from ml_toolkit.models import CatBoostClassificationModel
from ml_toolkit.training import train_model
from ml_toolkit.training.contracts import DatasetSplit, PostFitEvaluator


class CampaignEvaluator:
    def __init__(self, purchase_value_by_id: dict[int, float], contact_cost: float):
        self.purchase_value_by_id = purchase_value_by_id
        self.contact_cost = contact_cost

    def evaluate(
        self, *, object_id: pl.Series, y_pred: np.ndarray,
        y_true: np.ndarray, split: DatasetSplit,
    ) -> dict[str, float]:
        value = np.array([self.purchase_value_by_id[i] for i in object_id.to_list()])
        selected = y_pred >= 0.7
        return {
            "expected_profit": float(
                (y_pred[selected] * value[selected] - self.contact_cost).sum()
            ),
            "realized_profit": float(
                (y_true[selected] * value[selected] - self.contact_cost).sum()
            ),
        }


# data is a Pool with binary targets and object_id populated.
# The reference dictionary must contain every ID in data.
evaluator: PostFitEvaluator = CampaignEvaluator(purchase_value_by_id, contact_cost=10.0)
result = train_model(
    data=data,
    params={"iterations": 200, "loss_function": "Logloss", "verbose": False},
    model_factory=CatBoostClassificationModel,
    metric=metrics_classification,
    post_fit_evaluator=evaluator,
    run_name="campaign", n_splits=3,
)
```

Each call receives one complete split, with aligned IDs, predictions, and
targets in row order. Predictions follow `predict_pool`: positive-class
probabilities for binary classification, target predictions for regression.
`DatasetSplit` has exactly `TRAIN` ("train"), `VAL` ("val"), and `TEST` ("test").
The current training workflow has no independent test input, so it invokes only
train and val. To skip a split, return `{}`; for example, return `{}` whenever
`split != DatasetSplit.VAL` for validation-only evaluation.

The result must be a dictionary with nonempty string names and finite Python
or NumPy floating scalars. Integers, booleans, arrays, NaN, and infinities are
rejected. Names may contain word characters, spaces, periods, dashes, and
slashes, but no empty or relative path segments. The toolkit adds the prefix:
`train/custom/expected_profit`, `val/custom/expected_profit`, and so on. Results
are logged to each fold's MLflow run; they are not merged into `fold_metrics`,
averaged into `cv/*`, or made available to `metric_to_optimize`.

The same evaluator instance is reused, without copying or resetting, for every
CV fold and every Optuna trial, any repeated CV for diagnostics, and the final
fit. The final fit evaluates train only and adds one prediction pass when the
evaluator is enabled. Intermediate feature-selection fits do not invoke it.
Stored reference data and caches are supported; metrics should describe the
current inputs rather than accumulate unrelated folds or trials. Do not mutate
the input series or arrays. Evaluator errors propagate with stage and split
context attached to the traceback.

`Pool.object_id` is optional and keyword-only. Supplied IDs must be aligned,
non-null strings or integers; repeated IDs are allowed. Their original values
and dtype are preserved by row/feature/target operations, and IDs are not added
to model inputs. Supplying an evaluator without IDs fails before fitting.
Existing custom containers need no changes unless using this feature; for it,
implement `IdentifiedTrainingData`, including ID-preserving `take`,
`select_features`, and `with_target` operations.

### Custom post-fit images

Optionally add `write_artifacts` to the same evaluator object. It receives the
same `object_id`, `y_pred`, `y_true`, and `split`, plus a fresh temporary
`artifact_dir: Path`. Return `None` or `[]` to skip upload, or return a list of
existing `Path` objects selecting images saved inside that directory. Absolute
paths and paths relative to `artifact_dir` are supported. Unlisted files are
ignored; tables and other non-image files are rejected if returned.

For example, add this method to `CampaignEvaluator`:

```python
from pathlib import Path
import matplotlib.pyplot as plt
from ml_toolkit.plots.utils import save_figure

def write_artifacts(
    self, *, object_id: pl.Series, y_pred: np.ndarray,
    y_true: np.ndarray, split: DatasetSplit, artifact_dir: Path,
) -> list[Path] | None:
    if split != DatasetSplit.VAL:
        return None
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(y_pred, bins=20)
    ax.set(title="Validation scores", xlabel="Score", ylabel="Count")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return [save_figure(fig, artifact_dir / "figures/scores.png")]
```

No extra `train_model` option is needed. The method is called immediately after
each scalar evaluation, including Optuna trials, repeated CV, and final fitting,
even when `evaluate` returns `{}`. Existing evaluators without this method keep
their behavior. Each method can prepare its data independently; no cache is
required. Image generation is independent of `DiagnosticsOptions`, and the
writer decides whether to render for a given split. There is no automatic
refitting to produce images.

Only selected images are uploaded to `custom/<split>/`, preserving nested
directories; for example, `custom/val/figures/scores.png`. PNG, JPEG, SVG, WebP,
GIF, TIFF, and BMP extensions are supported. Temporary directories are removed
after upload or on errors. Use the exported `PostFitArtifactWriter` protocol
for the optional method's type contract; explicit inheritance is unnecessary.

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
| `training/contracts.py` | Training data and post-fit evaluator protocols; dataset split enum. |
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
