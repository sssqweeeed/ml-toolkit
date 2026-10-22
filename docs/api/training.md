# Training

Use `train_model` when you want chronological validation, optional selection and
tuning, a final fit, and MLflow reporting in one call.

## Run a holdout experiment

Run from an installed source checkout after `python -m examples.prepare_data`.
The example uses all 17,379 hourly [UCI Bike Sharing observations](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset),
with original 2011–2012 timestamps, eleven weather/calendar features, rental
counts as targets and unit weights per recorded hour. Count components that
would reveal the target are excluded from the inputs.

<!-- example: training-holdout -->

```python
from ml_toolkit.training import Holdout
import mlflow

from examples.gallery_data import load_bike_pool
from ml_toolkit import train_model
from ml_toolkit.config import TrackingConfig, TrainingConfig
from ml_toolkit.metrics import metrics_frequency
from ml_toolkit.models import CatBoostLogLinkRegressionModel

data = load_bike_pool(download_if_missing=False)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("ml-toolkit-examples")

params = {
    "iterations": 40,
    "depth": 4,
    "loss_function": "Poisson",
    "random_seed": 42,
    "thread_count": 2,
    "verbose": False,
    "allow_writing_files": False,
}
splitter = Holdout(data, val_size=0.25, test_size=0.2)
config = TrainingConfig(tracking=TrackingConfig(log_final_model=False))
result = train_model(
    splitter=splitter,
    params=params,
    model_factory=CatBoostLogLinkRegressionModel,
    metric=metrics_frequency,
    run_name="frequency-holdout",
    config=config,
)
print(
    "Validation Poisson deviance:",
    round(result.validation.mean_metrics["mean_poisson_deviance"], 4),
)
assert result.test_metrics is not None
print("Test Poisson deviance:", round(result.test_metrics["mean_poisson_deviance"], 4))
```

<!-- output: training-holdout -->
```text
Validation Poisson deviance: 118.8938
Test Poisson deviance: 111.7045
```
<!-- /output -->

The latest 20% of observations are reserved for test. The last 25% of the
remaining observations form validation; fractions round upward. After
validation, the final model fits all non-test rows and evaluates the future
test tail. Neither selection nor tuning sees test rows. The call creates a local `mlflow.db` and logs metrics and
configuration; model packaging is disabled in this example.

## Supply validation and final-fit data

`Holdout` and `TimeSeriesCV` bind a prepared pool to a splitting strategy.
`get_validation()` returns a fresh iterator of `Split(train, heldout, index)`;
`get_final()` returns `Split(train, heldout)` with an optional test sample.
Pools retain row metadata and their concrete container type.

<!-- example: training-splitter-data -->

```python
from ml_toolkit.training import TimeSeriesCV

cv = TimeSeriesCV(data, n_splits=3, test_size=0.2, gap=0, max_train_size=None)
for part in cv.get_validation():
    assert part.index is not None and part.heldout is not None
final = cv.get_final()
assert final.heldout is not None
```

Test fractions use the original pool; validation fractions use its non-test rows.
Fractions round upward. A test start date includes that calendar day in the
input timezone. `gap` omits rows before validation, and `max_train_size` bounds
CV training windows. Built-in final fitting includes every non-test row.
`skip_splits` omits initial CV windows while preserving their one-based indices.

The workflow consumes validation pairs in iterator order and requests final data
after validation. At least one validation pair is required, with positive unique
indices. Each pool needs ascending timestamps and valid targets and weights;
feature schemas and declarations must match within a pair. The splitter owns
sample membership and leakage prevention; object IDs can repeat.

## Use the result

`result.validation.mean_metrics` summarizes held-out validation scores;
`result.test_metrics` describes the future test tail. `result.model` is the final
fitted adapter. A single holdout produces one entry in
`result.validation.splits`.

To evaluate several chronological windows, pass
`splitter=TimeSeriesCV(data, n_splits=3, test_size=0.2)`. Each window performs its own selection and
tuning when enabled. Final fitting uses the last window's settings. See
[configuration by task](../configuration.md) for reporting and task-specific
settings.

## Select features and tune parameters

Continue after the holdout example. RFE selection needs a ranker and tuning needs
an Optuna search-space callback. This example compares all eleven features
against a three-feature subset, then runs two parameter trials. Small search
budgets demonstrate the workflow rather than a tuned benchmark.

<!-- example: training-search -->

```python
from dataclasses import replace

from optuna import Trial

from ml_toolkit.config import TuningConfig
from ml_toolkit.feature_selection import RFESelector, TopKSelector, ForwardSelector
from ml_toolkit.training import importance_ranker
from ml_toolkit.types import ModelParams


def param_space(trial: Trial) -> ModelParams:
    return {"depth": trial.suggest_int("depth", 1, 3)}


search_config = replace(
    config,
    tuning=TuningConfig(
        metric="mean_poisson_deviance",
        direction="minimize",
        n_trials=2,
        seed=42,
    ),
)
searched = train_model(
    splitter=splitter,
    params=params,
    model_factory=CatBoostLogLinkRegressionModel,
    metric=metrics_frequency,
    run_name="frequency-search",
    config=search_config,
    feature_selector=RFESelector(
        min_features=3,
        metric="mean_poisson_deviance",
        direction="minimize",
        max_fits=2,
        ranker=importance_ranker,
    ),
    param_space=param_space,
)
assert searched.feature_selection is not None and searched.tuning is not None
assert searched.test_metrics is not None
print("Selected features:", searched.features)
print(
    "Visited subset sizes:",
    [len(step.features) for step in searched.feature_selection.steps],
)
print("Best depth:", searched.tuning.best_params["depth"])
print(
    "Independent test deviance:",
    round(searched.test_metrics["mean_poisson_deviance"], 4),
)
```

<!-- output: training-search -->
```text
Selected features: ('temperature', 'feels_like_temperature', 'humidity', 'wind_speed', 'hour', 'working_day', 'weekday', 'holiday', 'season', 'weather', 'month')
Visited subset sizes: [11, 3]
Best depth: 3
Independent test deviance: 121.16
```
<!-- /output -->

The metric name must be a key returned by `metric(...).to_dict()`. Selection
and tuning use validation rows, so their best scores are search results. Use
`searched.test_metrics` for the independent future test evaluation. Each CV
window repeats this process; the final fit reuses the last window's settings.
`ShapRanker()` is an alternative ranker using native tree SHAP contributions;
it does not require the `shap` extra.

For an explicit train/validation pair, use `next(splitter.get_validation())` and
`selector.select(train=..., validation=..., params=..., model_factory=..., metric=...)` directly. Selection returns history and column names; apply
them with `data.select_features(selection.features)` before your own fit.
`tune_optuna` also works independently when supplied a scalar `score` callback.

## Tune outside the workflow

This standalone callback fits each trial on earlier development rows and
returns only its validation score. The reserved test tail is not passed in.

<!-- example: training-standalone-tuning -->

```python
from ml_toolkit.training import tune_optuna

partition = next(splitter.get_validation())


def score(candidate_params: ModelParams) -> float:
    candidate = CatBoostLogLinkRegressionModel(candidate_params).fit(
        partition.train, partition.heldout
    )
    return metrics_frequency(
        partition.heldout.y.to_numpy(),
        candidate.predict_pool(partition.heldout),
        partition.heldout.weight.to_numpy(),
    ).mean_poisson_deviance


tuned = tune_optuna(
    base_params=params,
    param_space=param_space,
    score=score,
    config=TuningConfig(
        metric="mean_poisson_deviance", direction="minimize", n_trials=2, seed=42
    ),
)
print("Best depth:", tuned.best_params["depth"])
print("Validation deviance:", round(tuned.best_score, 4))
```

<!-- output: training-standalone-tuning -->
```text
Best depth: 3
Validation deviance: 129.3511
```
<!-- /output -->

## Choose a selector

All selectors below work with the prepared `data`, `params`, model factory,
metric and `search_config` above. The selector owns its settings and ranker.
Omit `feature_selector` to
keep all columns. Each validation window uses fresh models, and the final fit
reuses the last window's selected columns.

<!-- example: training-selectors -->

```python
partition = next(splitter.get_validation())
selectors = (
    RFESelector(
        min_features=3,
        metric="mean_poisson_deviance",
        direction="minimize",
        ranker=importance_ranker,
        max_fits=2,
    ),
    TopKSelector(k=3, ranker=importance_ranker),
    ForwardSelector(
        metric="mean_poisson_deviance",
        direction="minimize",
        max_features=2,
        max_fits=21,
    ),
)
for selector in selectors:
    selection = selector.select(
        train=partition.train,
        validation=partition.heldout,
        params=params,
        model_factory=CatBoostLogLinkRegressionModel,
        metric=metrics_frequency,
    )
    print(type(selector).__name__, selection.features, len(selection.steps))
```

<!-- output: training-selectors -->
```text
RFESelector ('temperature', 'feels_like_temperature', 'humidity', 'wind_speed', 'hour', 'working_day', 'weekday', 'holiday', 'season', 'weather', 'month') 2
TopKSelector ('temperature', 'feels_like_temperature', 'hour') 2
ForwardSelector ('feels_like_temperature', 'hour') 21
```
<!-- /output -->

RFE repeatedly removes low-ranked columns and chooses the smallest visited
subset within `score_tolerance` of the best result. Top-K ranks once and keeps
exactly `k`, including protected features, even if its quality is worse. Forward
compares every remaining single-feature addition and accepts it only when the
absolute gain exceeds `min_improvement` (default zero). It does not begin a
round that exceeds `max_fits`; too little budget for its first comparison is an
error. With no protected columns it starts by comparing every singleton.

`protected_features` are mandatory and count toward every size limit. Ties
prefer the original feature order. `steps` contains all evaluated candidates,
including rejected ones, with metrics and timings; `selected_step` identifies
the returned subset. No fitted models are retained. `settings` records concrete
JSON-compatible configuration; executable rankers are excluded.
Its public type, `ml_toolkit.feature_selection.SelectionSettings`, is a flat
dictionary with string keys and values of type `str`, `int`, `float`, `bool`,
`None`, or `tuple[str, ...]`. Use it for a custom selector's configuration;
the tuple form records feature names.

Each validation run stores concrete settings and every evaluated subset in
`selection/history.json`, including scores, timings, available importances and
`selected_step`.
The selected scores are logged under `selection/metrics/<name>`; fit count,
feature count and selected index have separate `selection/` entries.

## Implement a selector

Implement `FeatureSelector.select` with the documented keyword arguments and
return a `FeatureSelectionResult`. The workflow checks the selected history
entry and input-column order. Create a fresh model for every fit, copy parameter
dictionaries, and keep learned state local to each call. The same selector
instance is reused across splits; it must not accumulate fitted estimators.

This reusable example evaluates a project-approved feature subset:

??? note "Complete custom selector source"

    ```python
    --8<-- "examples/custom_selector.py"
    ```

<!-- example: training-custom-selector -->

```python
from examples.custom_selector import FixedFeatureSelector

custom_selection = train_model(
    splitter=splitter,
    params=params,
    model_factory=CatBoostLogLinkRegressionModel,
    metric=metrics_frequency,
    run_name="custom-selector",
    config=config,
    feature_selector=FixedFeatureSelector(("temperature", "hour")),
)
print(custom_selection.features)
assert custom_selection.feature_selection is not None
print(custom_selection.feature_selection.settings)
```

<!-- output: training-custom-selector -->
```text
('temperature', 'hour')
{'selector': 'fixed', 'features': ('temperature', 'hour')}
```
<!-- /output -->

The normal workflow performs validation, optional tuning and a fresh final fit
using those columns. Prediction, refitting and model packaging remain the model
adapter's responsibility: serve inputs in `result.features` order through that
adapter. The selector itself owns no learned model state and is not a serving
artifact. See [model packaging](tracking.md) for saving and reloading the final
model. Custom selectors must use the supported `Pool` and `Model` contracts;
they do not extend backend or prediction-shape support.

The selected final model can be packaged and checked independently of the
selector. Continue with `custom_selection` above; this creates an MLflow model
artifact and verifies its reloaded predictions on the selected input schema.

<!-- example: training-selector-serving -->

```python
import numpy as np

development = splitter.get_final()
assert development.heldout is not None
selected_train = development.train.select_features(custom_selection.features)
selected_test = development.heldout.select_features(custom_selection.features)
expected = custom_selection.model.predict_pool(selected_test)
with mlflow.start_run():
    logged = custom_selection.model.log(selected_train)
model_info = logged[1] if isinstance(logged, tuple) else logged
reloaded = mlflow.pyfunc.load_model(model_info.model_uri)
np.testing.assert_allclose(reloaded.predict(selected_test.to_model_input()), expected)
print("Selected model predictions preserved after reload")
```

<!-- output: training-selector-serving -->
```text
Selected model predictions preserved after reload
```
<!-- /output -->

## Add custom evaluation

Import `PredictionEvaluator` from `ml_toolkit.evaluation`. Each call receives
one fit's combined pool, predictions and an aligned String `split` series.
IDs are optional. Continue with `data`,
`params`, and `config` from the holdout example.

<!-- example: training-custom-evaluation -->

```python
from pathlib import Path
import numpy as np
import polars as pl
from ml_toolkit.config import FigureConfig
from ml_toolkit.data import Pool
from ml_toolkit.evaluation import PredictionEvaluator
from ml_toolkit.feature_levels import FeatureLevelConfig
from ml_toolkit.types import FloatArray


class ExtraMetrics(PredictionEvaluator):
    def evaluate(
        self, *, data: Pool, prediction: FloatArray, split: pl.Series
    ) -> dict[str, float]:
        result = {}
        for part in split.unique().to_list():
            mask = (split == part).to_numpy()
            result[f"{part}/mae"] = float(
                np.mean(np.abs(data.y.to_numpy()[mask] - prediction[mask]))
            )
        return result

    def write_artifacts(
        self,
        *,
        data: Pool,
        prediction: FloatArray,
        split: pl.Series,
        artifact_dir: Path,
        figure: FigureConfig,
        features: tuple[FeatureLevelConfig, ...] | None,
        n_bins: int,
    ) -> list[Path]:
        heldout = split.filter(split != "train")
        if heldout.is_empty():
            return []
        part = heldout[0]
        mask = (split == part).to_numpy()
        directory = artifact_dir / part / "tables"
        directory.mkdir(parents=True)
        table = pl.DataFrame(
            {"observed": data.y.to_numpy()[mask], "predicted": prediction[mask]}
        )
        files = [
            directory / f"predictions.{extension}"
            for extension in ("csv", "parquet", "json", "html")
        ]
        table.write_csv(files[0])
        table.write_parquet(files[1])
        table.write_json(files[2])
        files[3].write_text(table.to_pandas().to_html(index=False))
        return files


evaluated = train_model(
    splitter=splitter,
    params=params,
    model_factory=CatBoostLogLinkRegressionModel,
    metric=metrics_frequency,
    run_name="frequency-custom-evaluation",
    config=config,
    prediction_evaluators=[ExtraMetrics(name="business"), ExtraMetrics(name="quality")],
)
client = mlflow.MlflowClient()
parent_run = client.search_runs(
    [mlflow.get_experiment_by_name("ml-toolkit-examples").experiment_id],
    "tags.mlflow.runName = 'frequency-custom-evaluation'",
)[0]
children = client.search_runs(
    [parent_run.info.experiment_id],
    f"tags.mlflow.parentRunId = '{parent_run.info.run_id}'",
)
final_run = next(
    run for run in children if run.data.tags["mlflow.runName"] == "final_fit"
)
print("Custom test MAE:", round(final_run.data.metrics["business/test/mae"], 3))
artifacts = client.list_artifacts(final_run.info.run_id, "business/test/tables")
print("Saved prediction files:", [Path(item.path).name for item in artifacts])
```

<!-- output: training-custom-evaluation -->
```text
Custom test MAE: 121.534
Saved prediction files: ['predictions.csv', 'predictions.html', 'predictions.json', 'predictions.parquet']
```
<!-- /output -->

Both methods have empty default implementations, so a report may implement only
metrics or artifacts. A single instance or a list is accepted; `None` and `[]`
disable custom calls. `ReportingConfig` stage flags gate both methods and PSI/SHAP.
Instances are reused, but bundled reports retain settings only. Inputs must not
be mutated. Returned metrics do not become CV averages or optimization objectives.

A call accepts train + val, train + test, or train only; val and test together
raise `ValueError`. Single-sample reports and SHAP use `val/` or `test/`
subfolders. See [comparison reports](../configuration.md#shared-psi-and-observedfitted-diagnostics)
for their plots and artifact paths.

Names must be unique ignoring case and contain ASCII letters, digits,
underscores, hyphens or dots, starting with a letter or digit. Enabled PSI/SHAP
reserve their names. Metrics use `<name>/<key>`; files use `<name>/<relative path>`.
Only explicitly returned files are uploaded; temporary directories are removed
on success and failure. Supported files include PNG, SVG, PDF, CSV, Parquet,
JSON and HTML. Errors propagate with evaluator and stage context.

Override `parameters()` to return JSON-compatible settings for experiment logging.

## Implement a splitter

Subclass `Splitter` to define project-specific windows. This example takes
explicit row boundaries in the prepared chronological pool:

??? note "Complete splitter source"

    ```python
    --8<-- "examples/custom_splitter.py"
    ```

Continue with `data`, `params` and `config` from the holdout example:

<!-- example: training-custom-splitter -->

```python
from examples.custom_splitter import ProjectSplitter

project_splitter = ProjectSplitter(
    data,
    validation_start=len(data) // 2,
    test_start=3 * len(data) // 4,
)
project_result = train_model(
    splitter=project_splitter,
    params=params,
    model_factory=CatBoostLogLinkRegressionModel,
    metric=metrics_frequency,
    run_name="project-splits",
    config=config,
)
assert len(project_result.validation.splits) == 1
assert project_result.test_metrics is not None
```

Each `get_validation()` call must return a fresh iterator. The last yielded
split supplies final-fit settings, while `get_final().train` determines the
final training rows. Return `heldout=None` to omit test evaluation. Final pools
must contain the selected features. `parameters()` optionally supplies explicit
JSON-compatible settings logged with the class name in
`configuration/splitter.json`; it defaults to `{}`.

## API reference

Import selectors and selection results from `ml_toolkit.feature_selection`;
import workflow operations from `ml_toolkit.training`.
`train_model` and `TrainingResult` are also available from `ml_toolkit`.

### Training workflow

::: ml_toolkit.training.train_model

### Feature selection

Rankers are called as `ranker(model, data)`. Their callable interfaces define
feature coverage and returned importance values.

::: ml_toolkit.feature_selection.FeatureSelector
    options:
      members: [select]

::: ml_toolkit.feature_selection.RFESelector
    options:
      members: [select]

::: ml_toolkit.feature_selection.TopKSelector
    options:
      members: [select]

::: ml_toolkit.feature_selection.ForwardSelector
    options:
      members: [select]

::: ml_toolkit.training.importance_ranker

::: ml_toolkit.training.ShapRanker

::: ml_toolkit.training.ShapRanker.__call__

::: ml_toolkit.training.FeatureRanker

::: ml_toolkit.training.FeatureRanker.__call__

::: ml_toolkit.feature_selection.FeatureSelectionResult

::: ml_toolkit.feature_selection.FeatureSelectionStep

### Parameter tuning

::: ml_toolkit.training.tune_optuna

::: ml_toolkit.training.TuningResult

### Chronological splits

::: ml_toolkit.training.Splitter
    options:
      merge_init_into_class: true
      members: [get_validation, get_final, parameters]

::: ml_toolkit.training.Split
    options:
      members: [train, heldout, index]

::: ml_toolkit.training.Holdout
    options:
      merge_init_into_class: true
      members: [get_validation, get_final, parameters]

::: ml_toolkit.training.TimeSeriesCV
    options:
      merge_init_into_class: true
      members: [get_validation, get_final, parameters]

### Workflow results

::: ml_toolkit.training.TrainingResult

::: ml_toolkit.training.ValidationResult

::: ml_toolkit.training.SplitResult

### Custom evaluation

::: ml_toolkit.evaluation.PredictionEvaluator
    options:
      merge_init_into_class: true
      members: [name, evaluate, write_artifacts]

::: ml_toolkit.training.DatasetSplit

### Ready-made reports

::: ml_toolkit.evaluation.PrecisionRecall

::: ml_toolkit.evaluation.F1Threshold

::: ml_toolkit.evaluation.Calibration

::: ml_toolkit.evaluation.Gains

::: ml_toolkit.evaluation.Lift

::: ml_toolkit.evaluation.PUScoreDistribution

::: ml_toolkit.evaluation.PUPositiveCapture

::: ml_toolkit.evaluation.ObservedFitted

::: ml_toolkit.evaluation.TimeObservedFitted

::: ml_toolkit.evaluation.NumericalFeatureDistributions
