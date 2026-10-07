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

from ml_toolkit.models import CatBoostLogLinkRegressionModel
from ml_toolkit.metrics import metrics_frequency
from ml_toolkit.config import (
    CrossValidationConfig, PUDiagnosticsConfig, RegressionDiagnosticsConfig,
    TrainingConfig,
)
from ml_toolkit.training import train_model
from my_project.data import prepare_frequency_data

mlflow.set_experiment("Frequency")
data = prepare_frequency_data(raw_data)  # Your project prepares and returns a Pool.
result = train_model(
    data=data,
    params={"iterations": 200, "loss_function": "Poisson", "verbose": False},
    model_factory=CatBoostLogLinkRegressionModel,
    metric=metrics_frequency,
    run_name="frequency",
    config=TrainingConfig(
        cv=CrossValidationConfig(n_splits=4),
        diagnostics=RegressionDiagnosticsConfig(task="frequency"),
    ),
)
print(result.cross_validation.mean_metrics)
```

`train_model` starts an MLflow run, performs time-series cross-validation,
optionally selects features and tunes parameters, fits a final model, logs it,
and returns the fitted model and metrics. Each fold logs metrics and, when
requested, PSI tables and figures. `diagnostics=None` disables diagnostics.
Task-specific diagnostics default to the last evaluated fold, with SHAP disabled.
Models are packaged only at final fit by default. See
[configuration by task](docs/configuration.md) for one fully expanded example
per supported task. Every configuration field has NumPy-style documentation
in `ml_toolkit.config`; these classes are also exported by `ml_toolkit.training`.
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
`metrics_classification`, and `ClassificationDiagnosticsConfig()`.
The same `train_model` function performs the workflow. Optuna tuning is enabled
with `TrainingConfig(tuning=TuningConfig(...), ...)` and a separately supplied
`param_space`. Set the metric name, direction, trial/time budget, and seed in
`TuningConfig`. Feature selection uses `RFEFeatureSelectionConfig`
inside `train_model`: batch RFE fits fresh adapters on shrinking feature subsets
using a fixed internal chronological train/validation split and optional row gap.
Pass `feature_ranker=importance_ranker` or `feature_ranker=ShapRanker(...)` from
`ml_toolkit.training`. Custom rankers accept a fitted adapter and validation data,
returning a finite importance for every current column. The algorithm never
invokes a backend's feature selector.

`metric_name` selects one value from the existing metric callable; `direction`
and absolute `score_tolerance` choose the smallest visited subset close enough
to the best score. `min_features` is a search floor, and `max_fits` limits all
selection fits, including the full set. Protected columns remain in every subset.
The full set can win. The selected columns retain input order and are reused for
CV, tuning and final fitting. Internal scores are selection results, not an
independent estimate of performance.

`result.feature_selection` contains the selected step and each subset's metrics,
importances, removed columns, timings and best iteration. Training logs this
history to `selection/history.json` and scalar selection summaries. Standalone
`ml_toolkit.training.select_features` returns the same result without starting
MLflow runs or invoking diagnostics, model packaging or post-fit evaluators.

`importance_ranker` uses CatBoost prediction-change importance, LightGBM gain,
or another estimator's importances/coefficients. `ShapRanker` uses weighted mean
absolute native raw-output SHAP on a fixed validation sample, handles null
categories, and requires no `shap` extra. This ranking explains the inner model;
subset quality always uses the complete adapter's predictions and row weights.

Custom adapters implement `fit`, `native_model`,
`predict_pool`, `best_iteration` and `log(sample, config=TrackingConfig)`. Final iteration budgets follow
`FinalFitConfig(iteration_policy="cv_median")`; use `"provided"` to retain yours.

```python
from ml_toolkit.config import RFEFeatureSelectionConfig
from ml_toolkit.training import ShapRanker

config = TrainingConfig(
    cv=CrossValidationConfig(n_splits=4),
    feature_selection=RFEFeatureSelectionConfig(
        min_features=10,
        metric_name="mean_poisson_deviance",
        direction="minimize",
        max_fits=6,
    ),
)
# Use prepared frequency data with at least ten features and the imports above.
result = train_model(
    data=data,
    params={"iterations": 200, "loss_function": "Poisson", "verbose": False},
    model_factory=CatBoostLogLinkRegressionModel,
    metric=metrics_frequency,
    feature_ranker=ShapRanker(max_samples=1000, seed=42),
    run_name="frequency-rfe",
    config=config,
)
print(result.feature_selection.selected_step, result.features)
```

### Model policies and serving

`CatBoostRegressionModel` accepts CatBoost's native losses and custom objective
objects without a toolkit whitelist. It keeps native initialization and
loss-dependent prediction behavior (`prediction_type=None`), including signed
RMSE outputs. `loss` aliases `loss_function`; conflicting values are rejected.
Multi-output native predictions keep their dimensions, while the shared
training workflow requires one finite prediction per row.

`CatBoostLogLinkRegressionModel` specializes regression for Poisson and Tweedie.
Its baseline is `log(weighted target mean)`, requiring a finite positive mean.
It disables `boost_from_average`, supplies the same baseline to training and
validation, then adds it to native bias once. Other initialization settings are
preserved by the general regressor. The public `baseline` is None for native
initialization and a float for the specialized regressor.

`CatBoostClassificationModel` supports binary Logloss/CrossEntropy and uses
clipped weighted prevalence log-odds. It selects an unweighted F1 threshold on
validation (highest cutoff on ties), or 0.5 without validation. Set
`decision_threshold` to disable tuning. `CatBoostPUClassificationModel` inherits
the shared fit and prediction operations and uses population-prior log-odds,
CPU uPU/nnPU training and an explicit threshold policy. Both classification
classes add their baseline once to raw scores before the sigmoid.

`InflationAdjustedModel` composes either regressor with an inflation strategy.
The strategy is fitted on training rows only; serving dates come from
`data.time` for pool inputs or `prediction_date_column` for frame inputs.

Full classes implement training and MLflow inference. `predict_frame` is the
common prediction path; `predict` and `predict_pool` delegate to it. Models From
Code exports contain a small entry point, package sources, pinned dependencies,
a native model file and version-1 JSON state. Loading restores the exact model
class, baseline policy, threshold, PU settings and nested inflation regressor.
Ordinary imports perform no MLflow registration. Custom objective objects are
passed unchanged during training; JSON records their type for audit and native
artifacts supply inference. Inflation logging emits both native and adjusted
models; only the adjusted artifact uses the configured registry name.

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
from ml_toolkit.config import CrossValidationConfig, TrainingConfig
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
    run_name="campaign",
    config=TrainingConfig(cv=CrossValidationConfig(n_splits=3)),
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
averaged into `cv/*`, or made available to the tuning objective.

The same evaluator instance is reused without copying or resetting.
`PostFitEvaluationConfig` enables CV and final fit by default; trials are excluded.
Use its `stages`, `splits`, and `write_artifacts` fields to control calls.
After tuning, custom CV evaluation uses a dedicated pass on the winning settings. The final fit evaluates train only and adds one prediction pass when the
evaluator is enabled. Intermediate feature-selection fits do not invoke it.
Stored reference data and caches are supported; metrics should describe the
current inputs rather than accumulate unrelated folds or trials. Do not mutate
the input series or arrays. Evaluator errors propagate with stage and split
context attached to the traceback.

`Pool.object_id` is optional and keyword-only. Supplied IDs must be aligned,
non-null strings or integers; repeated IDs are allowed. Their original values
and dtype are preserved by row/feature/target operations, and IDs are not added
to model inputs. Enabled evaluation without IDs fails before fitting.
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
each enabled scalar evaluation, even when `evaluate` returns `{}`.
Set `PostFitEvaluationConfig(write_artifacts=False)` to suppress image calls.
Existing evaluators without this method keep their behavior. Each method can
prepare its data independently; no cache is required. Custom image generation
is independent of built-in diagnostics. The post-fit stage/split policy and
writer decide when to render. After tuning, enabled custom CV evaluation runs
in a dedicated validation pass for the winning parameters.

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
from ml_toolkit.config import (
    CrossValidationConfig, PUDiagnosticsConfig, RegressionDiagnosticsConfig,
    TrainingConfig,
)
from ml_toolkit.training import train_model
from ml_toolkit.models import CatBoostPUClassificationModel

prior = 0.08  # Estimate from the target population, not the labeled fraction.
result = train_model(
    data=pu_data, model_factory=CatBoostPUClassificationModel,
    params={"loss": "nnPU", "pu_prior": prior, "iterations": 200,
            "depth": 5, "learning_rate": 0.05, "task_type": "CPU"},
    metric=PUMetric(prior, loss="nnPU"),
    config=TrainingConfig(
        cv=CrossValidationConfig(n_splits=3),
        diagnostics=PUDiagnosticsConfig(),
    ),
    run_name="pu",
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
their own output configuration. The temporary selection model uses the supplied
training parameters; nnPU's internal one-tree fits use
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
| `config.py` | Immutable, validated settings for workflow and task-specific plots. |
| `data.py` | Shared Pool container, aligned row operations, and model conversions. |
| `training/train.py` | Select, tune, validate, fit, and log one experiment. |
| `training/cross_validation.py` | Time-based folds, evaluation, and nested MLflow runs. |
| `training/feature_selection.py` | Adapter-driven batch RFE and full selection history. |
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
