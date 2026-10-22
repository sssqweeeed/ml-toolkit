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
metrics. The standalone callback itself chooses its scalar score.

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
