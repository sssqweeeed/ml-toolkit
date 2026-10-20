# Training

Use `train_model` when you want chronological validation, optional selection and
tuning, a final fit, and MLflow reporting in one call.

## Run a holdout experiment

Continue with the `data` pool from the [data example](data.md#prepare-a-pool).
The [getting started](../getting-started.md) page contains a standalone version
with a larger synthetic dataset.

<!-- example: training-holdout -->

```python
import mlflow

from ml_toolkit import train_model
from ml_toolkit.config import HoldoutConfig, TrackingConfig, TrainingConfig
from ml_toolkit.metrics import metrics_frequency
from ml_toolkit.models import CatBoostLogLinkRegressionModel

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("ml-toolkit-examples")

params = {
    "iterations": 20,
    "depth": 2,
    "loss_function": "Poisson",
    "random_seed": 42,
    "thread_count": 2,
    "verbose": False,
    "allow_writing_files": False,
}
config = TrainingConfig(
    validation=HoldoutConfig(val_size=0.25),
    test_size=0.2,
    tracking=TrackingConfig(log_final_model=False),
)
result = train_model(
    data=data,
    params=params,
    model_factory=CatBoostLogLinkRegressionModel,
    metric=metrics_frequency,
    run_name="frequency-holdout",
    config=config,
)
print(result.validation.mean_metrics)
print(result.test_metrics)
```

<!-- output: training-holdout -->
```text
{'mean_poisson_deviance': 1.1157166906742346, 'd2_poisson_score': -0.006024335420477422, 'mean_absolute_error': 0.8147537483416682, 'root_mean_squared_error': 0.8979581325982117, 'prediction_over_observed': 0.9288441980216964, 'lift_at_10': 0.0, 'gain_at_10': 0.0, 'normalized_gini': -0.3333333333333334}
{'mean_poisson_deviance': 0.7329916919680924, 'd2_poisson_score': -0.09163255520540625, 'mean_absolute_error': 0.6309718855189681, 'root_mean_squared_error': 0.7894992039293456, 'prediction_over_observed': 0.7902734633982873, 'lift_at_10': 1.6666666666666665, 'gain_at_10': 0.16666666666666666, 'normalized_gini': 0.1999999999999992}
```
<!-- /output -->

Of the 24 rows, the latest 5 become test. Of the remaining 19, the latest 5
become validation and the first 14 become training. Fractions round upward.
After validation, the final model fits all 19 non-test rows and evaluates the
5 test rows. The call creates a local `mlflow.db` and logs metrics and
configuration; model packaging is disabled in this example.

## Use the result

`result.validation.mean_metrics` summarizes held-out validation scores;
`result.test_metrics` describes the future test tail. `result.model` is the final
fitted adapter. A single holdout produces one entry in
`result.validation.splits`.

To evaluate several chronological windows, change `validation` to
`TimeSeriesCVConfig(n_splits=3)`. Each window performs its own selection and
tuning when enabled. Final fitting uses the last window's settings. See
[configuration by task](../configuration.md) for reporting and task-specific
settings.

## Select features and tune parameters

Continue after the holdout example. RFE selection needs a ranker and tuning needs
an Optuna search-space callback. This small example visits the two-feature and
one-feature subsets, then compares two parameter trials on the selected subset.

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
    data=data,
    params=params,
    model_factory=CatBoostLogLinkRegressionModel,
    metric=metrics_frequency,
    run_name="frequency-search",
    config=search_config,
    feature_selector=RFESelector(
        min_features=1, metric="mean_poisson_deviance", direction="minimize",
        max_fits=2, ranker=importance_ranker,
    ),
    param_space=param_space,
)
print(searched.features)
print(searched.feature_selection.selected_step)
print(searched.tuning.best_params)
```

<!-- output: training-search -->
```text
('region',)
1
{'iterations': 20, 'depth': 2, 'loss_function': 'Poisson', 'random_seed': 42, 'thread_count': 2, 'verbose': False, 'allow_writing_files': False}
```
<!-- /output -->

The metric name must be a key returned by `metric(...).to_dict()`. Selection
and tuning use validation rows, so their best scores are search results. Use
`searched.test_metrics` for the independent future test evaluation. Each CV
window repeats this process; the final fit reuses the last window's settings.
`ShapRanker()` is an alternative ranker using native tree SHAP contributions;
it does not require the `shap` extra.

For an explicit train/validation pair, use `split_holdout` and
`selector.select(train=..., validation=..., params=..., model_factory=..., metric=...)` directly. Selection returns history and column names; apply
them with `data.select_features(selection.features)` before your own fit.
`tune_optuna` also works independently when supplied a scalar `score` callback.

## Choose a selector

All selectors below work with the prepared `data`, `params`, model factory,
metric and `search_config` above. The selector owns its settings and ranker;
`TrainingConfig` has no feature-selection settings. Omit `feature_selector` to
keep all columns. Each validation window uses fresh models, and the final fit
reuses the last window's selected columns.

<!-- example: training-selectors -->

```python
from ml_toolkit.training import split_holdout

partition = split_holdout(data, HoldoutConfig(val_size=0.25))
selectors = (
    RFESelector(min_features=1, metric="mean_poisson_deviance", direction="minimize",
                ranker=importance_ranker, max_fits=2),
    TopKSelector(k=1, ranker=importance_ranker),
    ForwardSelector(metric="mean_poisson_deviance", direction="minimize",
                    max_features=2, max_fits=3),
)
for selector in selectors:
    selection = selector.select(
        train=partition.train, validation=partition.val, params=params,
        model_factory=CatBoostLogLinkRegressionModel, metric=metrics_frequency,
    )
    print(type(selector).__name__, selection.features, len(selection.steps))
```

<!-- output: training-selectors -->
```text
RFESelector ('age', 'region') 2
TopKSelector ('age',) 2
ForwardSelector ('region',) 3
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
`selected_step`. No fitted models or executable rankers are serialized there.
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
    data=data, params=params, model_factory=CatBoostLogLinkRegressionModel,
    metric=metrics_frequency, run_name="custom-selector", config=config,
    feature_selector=FixedFeatureSelector(("age",)),
)
print(custom_selection.features)
print(custom_selection.feature_selection.settings)
```

<!-- output: training-custom-selector -->
```text
('age',)
{'selector': 'fixed', 'features': ('age',)}
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

selected_data = data.select_features(custom_selection.features)
expected = custom_selection.model.predict_pool(selected_data)
with mlflow.start_run():
    logged = custom_selection.model.log(selected_data)
model_info = logged[1] if isinstance(logged, tuple) else logged
reloaded = mlflow.pyfunc.load_model(model_info.model_uri)
np.testing.assert_allclose(reloaded.predict(selected_data.to_model_input()), expected)
print("Selected model predictions preserved after reload")
```

<!-- output: training-selector-serving -->
```text
Selected model predictions preserved after reload
```
<!-- /output -->

## Add custom evaluation

Custom evaluation needs aligned IDs in `Pool.object_id`. Continue with `data`,
`params`, and `config` from the holdout example. Each evaluator needs a unique
name, used for its metric and artifact namespace. The example passes two
instances that both report unweighted MAE and write the same filenames;
their results are stored separately.

<!-- example: training-custom-evaluation -->

```python
from pathlib import Path
import numpy as np
import polars as pl

from ml_toolkit.data import Pool
from ml_toolkit.training import DatasetSplit, PredictionEvaluator
from ml_toolkit.types import FloatArray


class ExtraMetrics(PredictionEvaluator):
    def evaluate(
        self,
        *,
        object_id: pl.Series,
        y_pred: FloatArray,
        y_true: FloatArray,
        split: DatasetSplit,
    ) -> dict[str, float]:
        return {"mae": float(np.mean(np.abs(y_true - y_pred)))}

    def write_artifacts(
        self, *, object_id: pl.Series, y_pred: FloatArray, y_true: FloatArray,
        split: DatasetSplit, artifact_dir: Path,
    ) -> list[Path]:
        directory = artifact_dir / "tables"
        directory.mkdir()
        table = pl.DataFrame({"id": object_id, "observed": y_true, "predicted": y_pred})
        paths = [directory / f"predictions.{extension}"
                 for extension in ("csv", "parquet", "json", "html")]
        table.write_csv(paths[0])
        table.write_parquet(paths[1])
        table.write_json(paths[2])
        paths[3].write_text(table.to_pandas().to_html(index=False))
        return paths


identified = Pool(
    X=data.X,
    y=data.y,
    time=data.time,
    weight=data.weight,
    cat_features=data.cat_features,
    numerical_features=data.numerical_features,
    object_id=pl.Series("object_id", range(len(data))),
)
evaluated = train_model(
    data=identified,
    params=params,
    model_factory=CatBoostLogLinkRegressionModel,
    metric=metrics_frequency,
    run_name="frequency-custom-evaluation",
    config=config,
    prediction_evaluators=[
        ExtraMetrics(name="business"),
        ExtraMetrics(name="quality"),
    ],
)
```

With `ReportingConfig.prediction_evaluation=None`, supplying an evaluator uses the default
policy: available train, validation, and test rows during validation and final
fitting. Pass one evaluator directly, a list, or `None`; an empty list also
disables custom evaluation. All evaluators share the policy and run in list
order using the same predictions. Custom scores appear in MLflow under
`<split>/<name>/<metric>`, such as `val/business/mae` and `val/quality/mae`;
they do not enter validation averages or the tuning objective. Each instance
is reused, so return scores for the supplied rows and keep inputs unchanged.

Names start with an ASCII letter or digit and contain only ASCII letters,
digits, underscores, hyphens or dots. Names are checked for uniqueness without
regard to case before training, including when evaluation is disabled.
The `name` property is read-only. Subclasses with their own constructor must
call `super().__init__(name=name)`.

Use `ReportingConfig(prediction_evaluation=PredictionEvaluationConfig(...))` to choose stages
and splits, or empty stages/splits to disable calls. The workflow resolves these
calls before training and requires IDs only if a call is scheduled. For example,
a test-only policy needs no IDs when `test_size=None`, and a tuning-only policy
needs none when tuning is disabled.

To log images, CSV, Parquet, JSON, or HTML, override `write_artifacts`, save them inside the
supplied temporary `artifact_dir`, and return their paths. The method's reference
below describes supported files and directory lifetime. Files are stored under
`<name>/<split>/`, preserving your relative subdirectories. For example, both
`business/val/tables/predictions.csv` and `quality/val/tables/predictions.csv`
are retained. Each call receives its own temporary directory, and every returned
path is validated before any upload. An evaluator error stops the workflow with
its name, stage and split attached; earlier logged results remain available.

Existing evaluator implementations must now supply a name. Update consumers of
the previous `custom` paths and metric prefixes to use that name; historical
MLflow runs are unchanged.

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

::: ml_toolkit.training.split_holdout

::: ml_toolkit.training.make_validation_splits

::: ml_toolkit.training.TrainValSplit

### Workflow results

::: ml_toolkit.training.TrainingResult

::: ml_toolkit.training.ValidationResult

::: ml_toolkit.training.SplitResult

### Custom evaluation

::: ml_toolkit.training.PredictionEvaluator
    options:
      merge_init_into_class: true
      members: [name, evaluate, write_artifacts]

::: ml_toolkit.training.DatasetSplit
