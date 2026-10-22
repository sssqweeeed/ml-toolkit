# Feature importance and SHAP

Native feature importance summarizes how a fitted model uses its inputs. SHAP
explains the features' contribution to individual raw predictions. Neither
method establishes a causal effect.

## Prepare a fitted model

Run from the installed repository checkout after the [data preparation](index.md#example-data).
The example fits recorded 2011 hours and evaluates 2012 hours. The following figures explain the rental-count regressor on real
held-out hours. Native contributions work without the optional `shap` package;
the SHAP plots require that extra.

<!-- example: explainability-setup -->

```python
from pathlib import Path

import numpy as np
import polars as pl

from examples.evaluation_data import fit_bike_evaluation
from ml_toolkit.config import FigureConfig

example = fit_bike_evaluation(classification=False)
data, prediction, split = example.data, example.prediction, example.split
train, test = example.train, example.test
train_prediction = prediction[: len(train)]
test_prediction = prediction[len(train) :]
output = Path("figures")
figure = FigureConfig(dpi=150)
print(f"2011 train hours: {len(train)}; 2012 test hours: {len(test)}")
```

<!-- output: explainability-setup -->
```text
2011 train hours: 8645; 2012 test hours: 8734
```
<!-- /output -->

## Native feature importance

<!-- example: explainability-importance -->

```python
from ml_toolkit.plots import explainability

importance = example.model.feature_importances(train)
importance_table = explainability.feature_importance(
    list(importance),
    np.array(list(importance.values()), dtype=float),
    output / "feature-importance.png",
    max_features_display=11,
    figure=figure,
)
print(importance_table.head(6).round(3).to_string(index=False))
```

<!-- output: explainability-importance -->
```text
               feature  importance
                  hour      65.566
           working_day      11.751
           temperature       6.428
feels_like_temperature       5.513
                season       3.491
                 month       2.572
```
<!-- /output -->

![Native importance across the bike model's weather and calendar features.](../assets/plots/feature-importance.png)

The bars rank supplied importance values. Their scale depends on the backend's
importance definition. The function returns a pandas table sorted and limited
to displayed features; it accepts names and values, without requiring a model.

## Reconstruct individual raw predictions

<!-- example: explainability-contributions -->

```python
from ml_toolkit.explainability import shap_contributions

names, features, contributions = shap_contributions(
    example.model.native_model, test.X.head(3)
)
feature_effects = contributions[:, :-1]
base_values = contributions[:, -1]
reconstructed = base_values + feature_effects.sum(axis=1)
np.testing.assert_allclose(reconstructed, example.model.predict_pool(test)[:3])
print("Base rental count:", base_values.round(2).tolist())
print("Total feature effect:", feature_effects.sum(axis=1).round(2).tolist())
print("Reconstructed predictions:", reconstructed.round(2).tolist())
```

<!-- output: explainability-contributions -->
```text
Base rental count: [143.82, 143.82, 143.82]
Total feature effect: [-111.95, -129.17, -139.38]
Reconstructed predictions: [31.87, 14.65, 4.43]
```
<!-- /output -->

There is one contribution per feature followed by a base-value column.
Their sum reconstructs the native raw score. For this RMSE model, that is the
predicted rental count. For classifiers and log-link models the raw score
scale differs from the transformed probability or mean prediction.

## SHAP importance

<!-- example: explainability-shap -->

```python
from ml_toolkit.config import ShapConfig
from ml_toolkit.explainability import calculate_shap_values, top_shap_features

shap_config = ShapConfig(max_samples=600, max_display=11, seed=42)
explanation = calculate_shap_values(
    example.model.native_model, test.X, config=shap_config
)
explainability.shap_importance(
    explanation,
    output / "shap-importance.png",
    config=shap_config,
    figure=figure,
)
print("Contribution array:", explanation.values.shape)
print("Top three features:", top_shap_features(explanation, top_n=3))
```

<!-- output: explainability-shap -->
```text
Contribution array: (600, 11)
Top three features: ['hour', 'feels_like_temperature', 'temperature']
```
<!-- /output -->

![Mean absolute feature contributions over 600 sampled test hours.](../assets/plots/shap-importance.png)

The bars average absolute contributions, so they show magnitude, not direction.
Calculation samples at most `max_samples` rows with the supplied `seed`;
`max_display` limits visible features. Defaults are 10,000 samples and seed 0.
The explanation retains sampled feature values, raw contributions, and base values.

## SHAP beeswarm

<!-- example: explainability-beeswarm -->

```python
explainability.shap_beeswarm(
    explanation,
    output / "shap-beeswarm.png",
    config=shap_config,
    figure=figure,
)
```

![Distribution of feature contributions across sampled 2012 hours.](../assets/plots/shap-beeswarm.png)

Each point is one hour's contribution. Its horizontal position gives the signed
raw-score effect; color shows the feature value. Spread and color patterns show
how contributions differ between observations, beyond the average importance.

## SHAP dependence

<!-- example: explainability-dependence -->

```python
explainability.shap_dependence(
    explanation,
    "temperature",
    output / "shap-temperature.png",
    figure=figure,
)
```

![Temperature effects against normalized observed temperature, colored by interactions.](../assets/plots/shap-temperature.png)

The x-axis is the actual feature value and the y-axis its raw-score contribution.
Color can reveal a feature interaction. All three SHAP plotting calls return
`None`; reuse the same `shap.Explanation` for consistent sample comparisons.

## Enable SHAP during training

SHAP uses the fitted model, so configure it through reporting rather than a
prediction evaluator:

<!-- example: explainability-config -->

```python
from ml_toolkit.config import ReportingConfig

reporting = ReportingConfig(
    shap=ShapConfig(max_samples=600, max_display=11, seed=42)
)
```

Pass `TrainingConfig(reporting=reporting)` to the
[training workflow](../api/training.md). SHAP explains held-out rows and writes
`shap/val/importance.png` and `shap/val/beeswarm.png` during validation, with
`test/` for the final test run. Dependence plots are written for configured
features. A train-only final fit has no SHAP report. Shared stage flags and
figure settings are described in the [overview](index.md#reporting-inputs-and-outputs).

## API reference

::: ml_toolkit.explainability.shap_contributions
    options:
      heading_level: 3

::: ml_toolkit.explainability.calculate_shap_values
    options:
      heading_level: 3

::: ml_toolkit.explainability.top_shap_features
    options:
      heading_level: 3

::: ml_toolkit.plots.explainability
    options:
      heading_level: 3
      members: [feature_importance, shap_importance, shap_beeswarm, shap_dependence]
