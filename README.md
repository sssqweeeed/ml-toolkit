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

Projects prepare features and targets, order observations by ascending time, and
configure the MLflow URI and experiment. Pass prepared Polars data in `Pool` or a
custom object implementing `TrainingData`.

```python
from ml_toolkit import Pool

data = Pool(
    X=prepared.select(["age", "region"]),
    y=prepared["target"],
    time=prepared["date"],
    weight=prepared["weight"],
    cat_features=("region",),
    numerical_features=("age",),
    business_type=prepared["business_type"],  # Optional monthly reporting group.
    object_id=prepared["customer_id"],  # Optional; required for custom evaluation.
)
```

`take`, `select_features`, and `with_target` return new pools with aligned metadata.
Feature declarations may cover a subset of columns. IDs and business groups are
not automatically added to model inputs. `to_model_input` returns a pandas frame;
`to_catboost` includes labels, weights and timestamps. Both conversions cast
categories to strings and replace null categories with `"unknown"` without
changing the original frame.

```python
import mlflow
from ml_toolkit.config import (
    TrainingConfig, TimeSeriesCVConfig, HoldoutConfig, ReportingConfig,
    RegressionDiagnosticsConfig, RFEFeatureSelectionConfig, TuningConfig,
)
from ml_toolkit.models import CatBoostLogLinkRegressionModel
from ml_toolkit.metrics import metrics_frequency
from ml_toolkit.training import train_model, importance_ranker

mlflow.set_experiment("Frequency")
config = TrainingConfig(
    validation=TimeSeriesCVConfig(n_splits=4),
    test_size=0.2,
    feature_selection=RFEFeatureSelectionConfig(
        min_features=1, metric="mean_poisson_deviance", direction="minimize",
    ),
    reporting=ReportingConfig(
        diagnostics=RegressionDiagnosticsConfig(task="frequency"),
    ),
)
result = train_model(
    data=data,
    params={"iterations": 200, "loss_function": "Poisson", "verbose": False},
    model_factory=CatBoostLogLinkRegressionModel,
    metric=metrics_frequency,
    feature_ranker=importance_ranker,
    run_name="frequency",
    config=config,
)
print(result.validation.mean_metrics)
print(result.test_metrics)
```

`train_model` reserves an optional future test tail and prepares validation
splits. Each split independently selects features, tunes parameters, and fits the
winning settings once for reporting. Final fitting uses the last split's settings
on all non-test rows, then evaluates test. Replace `validation` with
`HoldoutConfig(val_size=0.2)` to execute the same process on one split.

Integers specify row counts; floats specify fractions in (0, 1), rounded up.
`test_size=None` disables test. `TimeSeriesCVConfig.val_size=None` uses the standard
`TimeSeriesSplit` block size. `gap` and `max_train_size` are rows, not durations.

To start test on a calendar date, use
`TrainingConfig(validation=HoldoutConfig(), test_size="2025-01-01")` or pass
`datetime.date(2025, 1, 1)`. Date strings must use `YYYY-MM-DD` format and are
normalized to dates when the configuration is created. Test includes every row
on that day and later, using the timezone of `data.time` for Datetime columns.
If that day has no observations, test starts on the next available day. The
cutoff must leave both training and test nonempty; exact datetimes are not accepted.

RFE runs once per split with the base model parameters. Candidates fit on that
split's train and use its val for scoring, early stopping and ranking. There is
no internal holdout. `min_features` is a search floor, `max_fits` is a fit budget,
and `score_tolerance` chooses the smallest visited subset close enough to the
best objective. Protected columns always remain. Selected features stay fixed
through that split's tuning and post-tuning fit.

Enable tuning with `TuningConfig` and an explicit `param_space` callback. Each
split starts its own Optuna study; trial and time budgets apply separately to
each study. Optuna consumes a scalar score callback and knows nothing about
holdout or TSCV. Each trial scores only the current split's val. TSCV's unweighted
mean of the post-tuning val metrics is a report, not an Optuna objective. Split
searches start from the same base parameters and do not inherit prior winners.

Final fitting takes the last split's selected features, winning parameters,
best iteration plus one and optional classification threshold. These settings
come from its post-tuning fit; without a best iteration, the original iteration
budget is retained. The final model includes former validation and gap rows,
performs no further RFE, and receives no validation or test during fitting.

`result.validation.splits` contains each split's train/val scores, selected
columns, refit settings, optional RFE history and optional `tuning` result,
without retaining fitted fold models. `result.feature_selection` and
`result.tuning` refer to the last split; `result.params` contains its final
constructor settings. `result.test_metrics` is absent when test is disabled.

Reports use the same functions for validation and test. `reporting.diagnostics=None`
disables built-in reports. `reporting.validation_splits` chooses `"last"`, `"all"`
or original one-based split indices; it does not restrict test reports or custom
evaluation. Built-in trial reports are disabled. SHAP defaults off.

Model packaging defaults to final fit only; enable validation and trial models
through `TrackingConfig`. Metrics and configuration artifacts are always logged.
Run names and metric prefixes use `validation`, with `train/*`, `val/*` and
`test/*` describing actual datasets. See [configuration by task](docs/configuration.md)
for fully expanded recipes.

Standalone `select_features(train=..., validation=..., ...)` performs the same
RFE on explicit datasets without logging or post-fit callbacks.
`importance_ranker`, `ShapRanker` and project-owned rankers remain available. `ShapRanker` uses weighted mean absolute raw-output contributions on a
fixed sample from the supplied validation rows and requires no `shap` extra. Diagnostic SHAP
explanations use the optional extra.

Custom adapters implement `fit`, `predict_pool`, `native_model`, `best_iteration`,
`feature_importances`, `get_refit_params` and `log`. `importance_ranker` delegates
to the adapter's explicit `feature_importances(data)` method.
`get_refit_params` returns copied constructor settings
for a fresh no-validation fit, incorporating selected iterations and thresholds
and disabling early stopping. Factories recreate nested models and strategies;
learned baselines and inflation state are fitted again on the final training rows.

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
passed unchanged during training; JSON uses their standard string representation for audit and native
artifacts supply inference. Inflation logging emits both native and adjusted
models; only the adjusted artifact uses the configured registry name.

### Metric contracts and type checking

Training metrics return an object with `to_dict() -> dict[str, float]`. Built-in
metric dataclasses implement this method. Wrap dictionary metrics explicitly:

```python
from ml_toolkit.metrics import MetricScores
from ml_toolkit.types import FloatArray


def mae(y: FloatArray, prediction: FloatArray, weight: FloatArray) -> MetricScores:
    error = abs(y - prediction)
    return MetricScores({"mae": float((error * weight).sum() / weight.sum())})
```

Pyright checks all of `src/ml_toolkit` in strict mode. Install the test extra and
run `python -m pyright` and `python -m pytest`. The checked-in `typings/` files
describe the third-party APIs used by the package; toolkit contracts use concrete
types and small protocols. Runtime checks cover data alignment, finite values,
weights, dates, feature names, mathematical ranges, and artifact paths.

### Custom post-fit evaluation

Use the separate `post_fit_evaluator` argument for business metrics that need
object IDs or externally prepared reference data. Subclass `PostFitEvaluator` and implement its `evaluate` method. Its constructor,
internal data, and helper methods are project-owned. The inherited
`write_artifacts` method returns an empty list.
The contracts live in `ml_toolkit.training.contracts` and are also exported by
`ml_toolkit.training`.

```python
import numpy as np
import polars as pl

from ml_toolkit.metrics import metrics_classification
from ml_toolkit.models import CatBoostClassificationModel
from ml_toolkit.config import TimeSeriesCVConfig, TrainingConfig
from ml_toolkit.training import train_model
from ml_toolkit.training.contracts import DatasetSplit, PostFitEvaluator
from ml_toolkit.types import FloatArray


class CampaignEvaluator(PostFitEvaluator):
    def __init__(self, purchase_value_by_id: dict[int, float], contact_cost: float):
        self.purchase_value_by_id = purchase_value_by_id
        self.contact_cost = contact_cost

    def evaluate(
        self, *, object_id: pl.Series, y_pred: FloatArray,
        y_true: FloatArray, split: DatasetSplit,
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
    config=TrainingConfig(validation=TimeSeriesCVConfig(n_splits=3)),
)
```

Each call receives one complete split, with aligned IDs, predictions, and
targets in row order. Predictions follow `predict_pool`: positive-class
probabilities for binary classification, target predictions for regression.
`DatasetSplit` has exactly `TRAIN` ("train"), `VAL` ("val"), and `TEST` ("test").
`test_size` enables test evaluation after final fitting. To skip a split, return
`{}`; for example, return `{}` whenever
`split != DatasetSplit.VAL` for validation-only evaluation.

The result must be a dictionary with nonempty string names and finite Python
or NumPy floating scalars. Integers, booleans, arrays, NaN, and infinities are
rejected. Names may contain word characters, spaces, periods, dashes, and
slashes, but no empty or relative path segments. The toolkit adds the prefix:
`train/custom/expected_profit`, `val/custom/expected_profit`, and so on. Results
are logged to each split's MLflow run; they are not merged into validation split
metrics,
averaged into `validation/*`, or made available to the tuning objective.

The same evaluator instance is reused without copying or resetting.
`PostFitEvaluationConfig` enables validation and final stages by default; trials
are excluded.
Use its `stages`, `splits`, and `write_artifacts` fields to control calls.
After its own tuning, each split evaluates the winning settings once. The final
stage evaluates train and optional test using the same predictions as ordinary
metrics. Intermediate feature-selection fits do not invoke it.
Stored reference data and caches are supported; metrics should describe the
current inputs rather than accumulate unrelated folds or trials. Do not mutate
the input series or arrays. Evaluator errors propagate with stage and split
context attached to the traceback.

`Pool.object_id` is optional and keyword-only. Supplied IDs must be aligned,
non-null strings or integers; repeated IDs are allowed. Their original values
and dtype are preserved by row/feature/target operations, and IDs are not added
to model inputs. Enabled evaluation without IDs fails before fitting.
Custom containers implement `TrainingData`, including optional `object_id`,
`business_type`, and `numerical_features` metadata. Their `take`,
`select_features`, and `with_target` operations preserve this metadata.

### Custom post-fit images

Optionally override `write_artifacts` in the same evaluator class. It receives the
same `object_id`, `y_pred`, `y_true`, and `split`, plus a fresh temporary
`artifact_dir: Path`. Return `[]` to skip upload, or return a list of
existing `Path` objects selecting images saved inside that directory. Absolute
paths and paths relative to `artifact_dir` are supported. Unlisted files are
ignored; tables and other non-image files are rejected if returned.

For example, add this method to `CampaignEvaluator`:

```python
from pathlib import Path
import matplotlib.pyplot as plt
from ml_toolkit.plots.utils import save_figure

def write_artifacts(
    self, *, object_id: pl.Series, y_pred: FloatArray,
    y_true: FloatArray, split: DatasetSplit, artifact_dir: Path,
) -> list[Path]:
    if split != DatasetSplit.VAL:
        return []
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
Evaluators that inherit the default writer skip image generation. Each method can
prepare its data independently; no cache is required. Custom image generation
is independent of built-in diagnostics. The post-fit stage/split policy and
writer decide when to render. Each split evaluates its winning parameters
once after tuning, without repeating RFE.

Only selected images are uploaded to `custom/<split>/`, preserving nested
directories; for example, `custom/val/figures/scores.png`. PNG, JPEG, SVG, WebP,
GIF, TIFF, and BMP extensions are supported. Temporary directories are removed
after upload or on errors. `PostFitEvaluator` declares both method contracts.

For positive–unlabeled learning, label `1` means confirmed positive and `0`
means unlabeled. Use a PU loss, a prevalence estimate, `PUMetric`, and PU
diagnostics together:

```python
from ml_toolkit.metrics import PUMetric
from ml_toolkit.config import (
    TimeSeriesCVConfig, PUDiagnosticsConfig, ReportingConfig, TrainingConfig,
)
from ml_toolkit.training import train_model
from ml_toolkit.models import CatBoostPUClassificationModel

prior = 0.08  # Estimate from the target population, not the labeled fraction.
result = train_model(
    data=pu_data,
    model_factory=CatBoostPUClassificationModel,
    params={
        "loss": "nnPU", "pu_prior": prior, "iterations": 200,
        "depth": 5, "learning_rate": 0.05, "task_type": "CPU",
    },
    metric=PUMetric(prior, loss="nnPU"),
    config=TrainingConfig(
        validation=TimeSeriesCVConfig(n_splits=3),
        reporting=ReportingConfig(diagnostics=PUDiagnosticsConfig()),
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
| `config/` | Immutable, validated settings for workflow and task-specific plots. |
| `data.py` | Shared Pool container, aligned row operations, and model conversions. |
| `training/workflow.py` | Prepare splits, run independent searches, average scores, and fit/evaluate final. |
| `training/splits.py` | Ready `TrainValSplit` datasets for holdout and temporal folds. |
| `training/fit.py` | RFE, independent Optuna, post-tuning evaluation for one split; final refit. |
| `training/feature_selection.py` | Adapter-driven batch RFE and full selection history. |
| `training/tuning.py` | Optuna parameter search using a scalar scoring callback. |
| `training/evaluation.py` | Shared train, validation, and test predictions and metrics. |
| `training/reporting.py` | PSI, plots, custom evaluation, and report artifacts. |
| `training/contracts.py` | Typed data/ranker contracts, evaluator base class, and dataset split enum. |
| `training/results.py` | Result dataclasses. |
| `models/` | CatBoost classification/regression, PU learning, LightGBM serving, and inflation-adjusted models. |
| `metrics/` | Classification, frequency, severity, and ranking calculations. |
| `psi/` | Fitted levels, PSI scoring, distributions, and results. |
| `plots/` | Every chart, including PSI charts. |
| `inflation.py` | Constant-rate and manually indexed inflation strategies. |
| `explainability.py` | SHAP extraction from supported native models. |
| `tracking/` | MLflow runs, configurations, tables, and model artifacts. |
| `tests/` | Behavioral and workflow integration tests. |

## Project boundaries

Project-specific input loading, joins, target definitions, Hydra runners,
inflation index values, and registered model names remain in their projects.
The toolkit accepts their prepared data and externally chosen settings.
