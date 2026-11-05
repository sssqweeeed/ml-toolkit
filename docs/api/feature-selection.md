# Feature selection

Selectors choose input columns using training and validation pools. A ranker
scores columns of an already fitted adapter; RFE and Top-K use that ordering.
The final model is fitted with the last validation split's selected columns.

## Prepare a selection experiment

Run from an installed checkout after `python -m examples.prepare_data --dataset bike`.
The imported `prepare_training()` loads the original Bike Sharing hours, model
parameters and local MLflow settings described in the [training example](training.md#run-a-holdout-experiment).
Targets are rentals per hour; weights are one per recorded hour.

<!-- example: selection-setup -->

```python
from examples.training_setup import prepare_training
from ml_toolkit import train_model
from ml_toolkit.metrics import metrics_frequency
from ml_toolkit.models import CatBoostLogLinkRegressionModel
from ml_toolkit.training import Holdout

data, params, config = prepare_training()
splitter = Holdout(data, val_size=0.25, test_size=0.2)
from ml_toolkit.feature_selection import (
    ForwardSelector,
    RFESelector,
    TopKSelector,
)
from ml_toolkit.training import importance_ranker
```

## Choose a selector

The selector owns its settings and ranker. Run the setup above first.
Omit `feature_selector` to
keep all columns. Each validation window uses fresh models, and the final fit
reuses the last window's selected columns.

<!-- example: selection-builtins -->

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
    print(
        type(selector).__name__,
        "features:",
        len(selection.features),
        "candidates:",
        len(selection.steps),
    )
```

<!-- output: selection-builtins -->
```text
RFESelector features: 11 candidates: 2
TopKSelector features: 3 candidates: 2
ForwardSelector features: 2 candidates: 21
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
JSON-compatible configuration; executable rankers and evaluators are excluded.
An evaluator-based selector records its evaluator's name under `evaluator`.
Its public type, `ml_toolkit.feature_selection.SelectionSettings`, is a flat
dictionary with string keys and values of type `str`, `int`, `float`, `bool`,
`None`, or `tuple[str, ...]`. Use it for a custom selector's configuration;
the tuple form records feature names.

Each validation run stores concrete settings and every evaluated subset in
`selection/history.json`, including scores, timings, available importances and
`selected_step`.
Ordinary validation scores remain in each step's `metrics`; an evaluator's
scores are stored separately in `objective_metrics`, or `None` when no evaluator
is configured. The selected ordinary scores are logged under
`selection/metrics/<name>`, and selected evaluator scores under
`selection/objective_metrics/<key>`; fit count,
feature count and selected index have separate `selection/` entries.

### Use an evaluator objective

RFE and Forward accept an optional `PredictionEvaluator` instance. `metric`
selects its exact returned key, without the evaluator's name. Omitting
`evaluator`, or setting it to `None`, uses the ordinary validation metric.
Top-K selects exactly `k` features from its ranker and has no evaluator option.

Continue with the setup above. The reusable
[custom evaluator](../evaluation/custom.md) computes weighted validation MAE;
the ordinary frequency metric continues to populate each candidate's `metrics`.

<!-- example: selection-evaluator -->

```python
from examples.custom_evaluator import PredictionAudit

audit = PredictionAudit(name="audit")
evaluator_selectors = (
    RFESelector(
        min_features=3,
        ranker=importance_ranker,
        evaluator=audit,
        metric="val/mae",
        direction="minimize",
        max_fits=3,
    ),
    ForwardSelector(
        evaluator=audit,
        metric="val/mae",
        direction="minimize",
        max_features=2,
        max_fits=21,
    ),
)
for selector in evaluator_selectors:
    selection = selector.select(
        train=partition.train,
        validation=partition.heldout,
        params=params,
        model_factory=CatBoostLogLinkRegressionModel,
        metric=metrics_frequency,
    )
    chosen = selection.steps[selection.selected_step]
    assert chosen.objective_metrics is not None
    print(
        type(selector).__name__,
        "features:", len(selection.features),
        "validation MAE:", round(chosen.objective_metrics["val/mae"], 3),
    )
```

<!-- output: selection-evaluator -->
```text
RFESelector features: 6 validation MAE: 128.957
ForwardSelector features: 2 validation MAE: 130.012
```
<!-- /output -->

Pass either configured selector through `train_model(feature_selector=...)` to
record its history and draw its search plots in the normal workflow. Plots use
`objective_metrics` when present and label the evaluator and metric. Ordinary
metrics, CV averages, and independent test scores keep their usual meanings.

For each candidate, the evaluator receives one combined train/validation pool,
aligned predictions, and `split` labels. It must select validation rows for its
objective; training rows can supply a comparison baseline. `data.X` contains the
candidate's selected features. Put metadata needed for every candidate in
`auxiliary_features`, which is retained along with targets, dates, weights, and
object IDs. Test rows are never passed into feature selection.

RFE uses the evaluator to choose the smallest visited subset within
`score_tolerance` of its best score; its ranker still controls elimination.
Forward uses it to compare additions and apply `min_improvement`, including a
protected-only baseline. Both thresholds are in the evaluator score's units.

Evaluator scoring runs independently of reporting flags. It calls only
`evaluate()`, once per fitted candidate, without generating evaluator artifacts.
Standalone selection starts no tracking runs and logs no metrics. Ordinary
selection predicts only validation rows; using an evaluator also predicts
training rows for each candidate, without additional model fits. The requested
key must exist and every evaluator score must be finite. Errors identify the
evaluator, requested key, and candidate feature subset. Evaluator scores stay
separate from ordinary scores even when their keys are identical.

### Automatic selection plots

`train_model` also writes diagnostic figures from each split's single search
history in that same `selection/` artifact folder:

| Selector | Artifact (default format) | Contents |
| --- | --- | --- |
| RFE | `selection/rfe_progress.png` | Validation objective against decreasing feature count, best and selected subsets, and the score tolerance band. |
| Top-K | `selection/top_k_ranking.png` | All full-model ranking values, with selected, excluded and protected features distinguished. |
| Forward | `selection/forward_progress.png` | All evaluated candidates by subset size, accepted additions, protected baseline when present, and the stopping reason. |
| Forward | `selection/forward_improvement.png` | The best candidate's improvement over the preceding accepted subset in each comparable round, with the strict `min_improvement` threshold. |
| All three | `selection/best_iteration.png` | Recorded best iteration against feature count, with the selected subset highlighted when its iteration is available. |

Forward improvement is positive when quality improves, for either optimization
direction. The initial singleton round without protected features establishes
the baseline; it has no improvement against an artificial zero score. With
protected features, the protected-only fit supplies the baseline. If there are
no comparable rounds, the improvement figure is skipped.

Best iterations are zero-based; zero is valid. Missing iterations are omitted
and break connecting lines instead of being plotted as zero. If every iteration
is missing, the figure is skipped. Forward displays all available candidates as
points, without connecting alternative subsets. RFE progress limits axis labels
on long schedules while retaining every evaluated point.

These search diagnostics are always generated when a built-in selector runs,
including when `ReportingConfig.on_validation=False`. They use the stored
history without additional fits, predictions or rankings. `ReportingConfig.figure`
controls size, DPI and format; SVG/PDF settings replace the filename extension.
Top-K has no before/after quality plot. Protected Top-K features can be below the
ranking cutoff, so a global Top-K line is shown only without protected features.
Selection scores describe the search, not independent performance.

Calling `selector.select()` directly only returns the result; it does not plot
or log artifacts. To draw an existing built-in result manually, use
`ml_toolkit.plots.feature_selection.rfe_progress(result, save_fig)`,
`top_k_ranking(result, save_fig)`, or `forward_progress(result, save_fig)`.
The same module provides `forward_improvement(result, save_fig)` and
`best_iteration_progress(result, save_fig)` for the additional diagnostics.
Each accepts an optional keyword-only `figure=FigureConfig(...)` and closes the
figure after saving. Custom selector identifiers are skipped by automatic
plotting; their history and metrics are still recorded as before.

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

<!-- example: selection-custom -->

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

<!-- output: selection-custom -->
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

<!-- example: selection-serving -->

```python
import numpy as np
import mlflow

development = splitter.get_test()
assert development.heldout is not None
selected_train = development.train.select_features(custom_selection.features)
selected_test = development.heldout.select_features(custom_selection.features)
expected = custom_selection.model.predict_pool(selected_test)
with mlflow.start_run():
    logged = custom_selection.model.log(selected_train)
model_info = logged[1] if isinstance(logged, tuple) else logged
reloaded = mlflow.pyfunc.load_model(model_info.model_uri)
np.testing.assert_allclose(
    reloaded.predict(selected_test.to_model_input(categorical_as_strings=True)),
    expected,
)
print("Selected model predictions preserved after reload")
```

<!-- output: selection-serving -->
```text
Selected model predictions preserved after reload
```
<!-- /output -->

## Choose a ranker

| Ranker | What its score means |
| --- | --- |
| `importance_ranker` | Importance supplied by the fitted adapter's native backend |
| `ShapRanker` | Weighted mean absolute native raw-output SHAP contribution |
| Custom callable | Your score for each current validation column |

`ShapRanker` supports CatBoost and LightGBM and does not require the `shap`
extra. It samples positive-weight validation rows deterministically. Native
raw outputs may have different units from adapter predictions.

<!-- example: selection-shap-ranker -->

```python
from ml_toolkit.training import ShapRanker

shap_selection = TopKSelector(
    k=3, ranker=ShapRanker(max_samples=500, seed=42)
).select(
    train=partition.train,
    validation=partition.heldout,
    params=params,
    model_factory=CatBoostLogLinkRegressionModel,
    metric=metrics_frequency,
)
print("SHAP-selected features:", shap_selection.features)
```

<!-- output: selection-shap-ranker -->
```text
SHAP-selected features: ('temperature', 'feels_like_temperature', 'hour')
```
<!-- /output -->

The selected names retain their original input order. SHAP affects ranking;
the candidate subset is still evaluated with the supplied task metric.

## Implement a ranker

A `FeatureRanker` is callable as `ranker(model, data)`. The model is already
fitted; `data` contains the current validation columns. Return exactly one
finite value per column name. Higher values rank higher, and signed values are
allowed. Preserve both inputs and retain no learned state between calls.

This custom ranker measures the increase in weighted validation MAE after
permuting one feature. A fresh seeded generator makes repeated calls
reproducible. Each perturbation starts from the original features, leaving
labels, weights, timestamps and IDs aligned:

??? note "Complete implementation"

    ```python
    --8<-- "examples/custom_ranker.py"
    ```

A high score means the fitted model relies on that feature for accurate
predictions. Correlated features can share predictive information, and a
single permutation is noisy; these scores are not causal effects. The
example ranks by MAE even though the workflow reports Poisson deviance.

<!-- example: selection-custom-ranker -->

```python
from examples.custom_ranker import PermutationMAERanker

ranked = train_model(
    splitter=splitter,
    params=params,
    model_factory=CatBoostLogLinkRegressionModel,
    metric=metrics_frequency,
    run_name="permutation-selection",
    config=config,
    feature_selector=TopKSelector(k=3, ranker=PermutationMAERanker(seed=42)),
)
print("Permutation-selected features:", ranked.features)
assert ranked.test_metrics is not None
print(
    "Test Poisson deviance:",
    round(ranked.test_metrics["mean_poisson_deviance"], 4),
)
```

<!-- output: selection-custom-ranker -->
```text
Permutation-selected features: ('temperature', 'feels_like_temperature', 'hour')
Test Poisson deviance: 112.9418
```
<!-- /output -->

`TopKSelector` creates the fitted model supplied to the ranker, applies the
returned ordering and evaluates the chosen subset. No model is fitted inside
the ranker. Explicit inheritance from `FeatureRanker` is unnecessary.

## API reference

### Selectors

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

### Rankers

::: ml_toolkit.training.importance_ranker

::: ml_toolkit.training.ShapRanker
    options:
      members: []

::: ml_toolkit.training.ShapRanker.__call__

::: ml_toolkit.training.FeatureRanker
    options:
      members: []

::: ml_toolkit.training.FeatureRanker.__call__

### Results

::: ml_toolkit.feature_selection.FeatureSelectionResult
    options:
      members: []

::: ml_toolkit.feature_selection.FeatureSelectionStep
    options:
      members: []

### Plots

::: ml_toolkit.plots.feature_selection
