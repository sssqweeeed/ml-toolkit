# PSI and feature levels

Population Stability Index (PSI) measures how feature distributions change
between a reference sample and a current sample. Fit categories and numerical
bins on the reference once, then reuse them for every comparison.

## Compare two samples

```python
import polars as pl

from ml_toolkit.config import FeatureLevelSpec
from ml_toolkit.psi import PSIAnalysis, PSIConfig

reference = pl.DataFrame(
    {"age": [20, 30, 40, 50], "region": ["north", "north", "south", "south"]}
)
current = pl.DataFrame(
    {"age": [45, 55, 65, 75], "region": ["north", "south", "west", "west"]}
)
analysis = PSIAnalysis(
    PSIConfig(
        features=(
            FeatureLevelSpec("age", "numerical", lower_bound=0),
            FeatureLevelSpec("region", "categorical"),
        ),
        n_bins=2,
    )
).fit(reference)
comparison = analysis.compare(current)
print(comparison.psi)
print(comparison.level_distribution)
```

`comparison.psi` has one score per feature. Larger values mean a greater change
in fitted-level shares. `level_distribution` shows the reference/current counts
and shares behind each score. The new `"west"` category uses the fitted unseen
level; numerical bins retain the reference boundaries.

Call `analysis.compare(next_sample)` for later batches. Re-fitting on each batch
would change the reference. For monthly distributions, provide aligned
`Date`/`Datetime` timestamps with `time=` in both `fit` and `compare`.

## Reuse feature levels

`LevelTransformer` exposes the same fit/transform operations when you need
consistent bins for another report. Continue with `reference` and `current`
above to transform both samples and compare their existing levels:

```python
from ml_toolkit.feature_levels import LevelTransformer
from ml_toolkit.psi import compare_psi

levels = LevelTransformer(analysis.config.features, n_bins=2).fit(reference)
reference_levels = levels.transform(reference)
current_levels = levels.transform(current)
assert levels.config is not None
level_comparison = compare_psi(reference_levels, current_levels, levels.config)
print(current_levels["region"].cast(pl.String).to_list())
# ['north', 'south', '__UNSEEN__', '__UNSEEN__']
```

Configured columns become Polars `Categorical` columns; other columns retain
their values and order. Reuse the fitted transformer for subsequent samples.
Training's shared PSI and observed/fitted diagnostics use these levels too; see the
[shared diagnostic recipe](../configuration.md#shared-psi-and-observedfitted-diagnostics).

## API reference

Import PSI analysis from `ml_toolkit.psi`.

::: ml_toolkit.psi
    options:
      members: [PSIAnalysis, PSIConfig, PSIResult, compare_psi, psi]

### Shared feature levels

Import these lower-level types from `ml_toolkit.feature_levels`.
`FeatureLevelSpec` is also exported from `ml_toolkit.config` and documented in the
[configuration reference](configuration.md).

::: ml_toolkit.feature_levels
    options:
      members:
        - LevelTransformer
        - LevelConfig
        - FeatureLevelConfig
        - FeatureType
        - FeatureScalar
        - NULL_LEVEL
        - ERROR_LEVEL
        - UNSEEN_LEVEL
