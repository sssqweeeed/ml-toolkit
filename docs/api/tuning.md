# Parameter tuning

Use an Optuna search-space callback with `train_model`, or call `tune_optuna`
with your own scalar scoring callback. Both paths create a `TuningResult` with
the best parameter dictionary, score and zero-based trial number.

## Tune in the training workflow

Run from an installed checkout after `python -m examples.prepare_data --dataset bike`.
The imported `prepare_training()` loads the original Bike Sharing hours, model
parameters and local MLflow settings described in the [training example](training.md#run-a-holdout-experiment).
Targets are rentals per hour; weights are one per recorded hour.

<!-- example: tuning-setup -->

```python
from examples.training_setup import prepare_training
from ml_toolkit import train_model
from ml_toolkit.metrics import metrics_frequency
from ml_toolkit.models import CatBoostLogLinkRegressionModel
from ml_toolkit.training import Holdout

data, params, config = prepare_training()
splitter = Holdout(data, val_size=0.25, test_size=0.2)
```

<!-- example: tuning-workflow -->

```python
from dataclasses import replace

from optuna import Trial

from ml_toolkit.config import TuningConfig
from ml_toolkit.feature_selection import RFESelector
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

<!-- output: tuning-workflow -->
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
See [feature selection](feature-selection.md) for selector and ranker options.

## Tune with an evaluator

Pass a `PredictionEvaluator` instance through `TuningConfig.evaluator` when the
objective needs pool metadata, auxiliary features, or a train/validation
comparison. `metric` selects the exact key returned by `evaluate()`; it does not
include the evaluator's name. Omitting `evaluator`, or setting it to `None`,
selects the ordinary validation metric as above.

This example reuses the setup and search space above and the
[custom evaluator](../evaluation/custom.md), which calculates weighted held-out
MAE. The ordinary `metric=metrics_frequency` still supplies training and test
scores; validation MAE guides this search.

<!-- example: tuning-evaluator -->

```python
from examples.custom_evaluator import PredictionAudit

audit = PredictionAudit(name="audit")
audit_config = replace(
    config,
    tuning=TuningConfig(
        evaluator=audit,
        metric="val/mae",
        direction="minimize",
        n_trials=2,
        seed=42,
    ),
    reporting=replace(config.reporting, on_tuning=False),
)
audited = train_model(
    splitter=splitter,
    params=params,
    model_factory=CatBoostLogLinkRegressionModel,
    metric=metrics_frequency,
    run_name="evaluator-search",
    config=audit_config,
    param_space=param_space,
)
assert audited.tuning is not None and audited.test_metrics is not None
print("Best depth:", audited.tuning.best_params["depth"])
print("Validation MAE:", round(audited.tuning.best_score, 4))
print("Independent test deviance:", round(audited.test_metrics["mean_poisson_deviance"], 4))
```

<!-- output: tuning-evaluator -->
```text
Best depth: 3
Validation MAE: 132.929
Independent test deviance: 121.16
```
<!-- /output -->

The evaluator receives combined train and validation rows, aligned predictions,
and split labels on every trial. Its implementation must select validation rows
for the objective; training rows can supply a comparison baseline. It must return
the requested key with a finite value on every trial and split. Missing keys,
invalid metric names, and nonfinite results raise errors with evaluator, metric,
split, and trial context. Test rows are never passed to the objective.

Objective-evaluator metrics are logged as `audit/<key>` in trial runs, even when
`on_tuning=False`; the selected scalar is also logged as `objective`. They do not
enter ordinary training results or CV averages. This evaluator runs only during
tuning and writes no artifacts. To also request reports, pass the same `audit`
instance through `prediction_evaluators`; reporting stage flags then control its
reports. When trial reporting is enabled, calculated scores are reused and
`evaluate()` runs once per trial. Different evaluator objects must have distinct
names, including the objective evaluator and all other report namespaces.

This option selects hyperparameters between trials. The backend training loss,
early stopping, and feature-selection metric retain their own settings.

## Tune outside the workflow

This standalone callback fits each trial on earlier development rows and
returns only its validation score. The reserved test tail is not passed in.

<!-- example: tuning-standalone -->

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
        metric="mean_poisson_deviance",
        direction="minimize",
        n_trials=2,
        seed=42,
    ),
)
print("Best depth:", tuned.best_params["depth"])
print("Validation deviance:", round(tuned.best_score, 4))
```

<!-- output: tuning-standalone -->
```text
Best depth: 3
Validation deviance: 129.3511
```
<!-- /output -->

The callback owns each trial's model and must return one finite score in the
configured direction. Copy or merge parameter dictionaries rather than retaining
a fitted model between trials. The search space receives `optuna.Trial`; use
its `suggest_*` methods to propose settings. `n_trials`, `timeout_seconds` and `seed`
control the budget and sampler; `metric` identifies the objective in workflow
metrics. The standalone callback itself chooses its scalar score; this optimizer
does not execute `config.evaluator`.

The result's `best_params` includes the base settings and winning suggestions;
`best_trial` is zero-based. Trial parameters and objective scores are logged in
nested MLflow runs.
Test data must remain outside both the search-space and scoring callbacks.

## API reference

See [configuration](configuration.md) for `TuningConfig` fields and defaults.

::: ml_toolkit.training.tune_optuna

::: ml_toolkit.training.TuningResult
    options:
      members: []
