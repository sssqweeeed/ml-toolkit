# PSI and feature levels

Population Stability Index (PSI) measures how feature distributions change
between a reference sample and a current sample. Fit categories and numerical
bins on the reference once, then reuse them for every comparison.

## Compare two samples

<!-- example: psi-compare -->

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
print(comparison.psi.sort("feature").with_columns(pl.col("psi").round(4)).to_dicts())
print(comparison.level_distribution.select("feature", "sample", "level", "rows").sort("feature", "sample", "level").to_dicts())
```

<!-- output: psi-compare -->
```text
[{'feature': 'age', 'psi': 6.9077}, {'feature': 'region', 'psi': 6.9077}]
[{'feature': 'age', 'sample': 'current', 'level': '[0, 35)', 'rows': 0}, {'feature': 'age', 'sample': 'current', 'level': '[35, inf)', 'rows': 4}, {'feature': 'age', 'sample': 'current', 'level': '__ERROR__', 'rows': 0}, {'feature': 'age', 'sample': 'current', 'level': '__NULL__', 'rows': 0}, {'feature': 'age', 'sample': 'reference', 'level': '[0, 35)', 'rows': 2}, {'feature': 'age', 'sample': 'reference', 'level': '[35, inf)', 'rows': 2}, {'feature': 'age', 'sample': 'reference', 'level': '__ERROR__', 'rows': 0}, {'feature': 'age', 'sample': 'reference', 'level': '__NULL__', 'rows': 0}, {'feature': 'region', 'sample': 'current', 'level': '__NULL__', 'rows': 0}, {'feature': 'region', 'sample': 'current', 'level': '__UNSEEN__', 'rows': 2}, {'feature': 'region', 'sample': 'current', 'level': 'north', 'rows': 1}, {'feature': 'region', 'sample': 'current', 'level': 'south', 'rows': 1}, {'feature': 'region', 'sample': 'reference', 'level': '__NULL__', 'rows': 0}, {'feature': 'region', 'sample': 'reference', 'level': '__UNSEEN__', 'rows': 0}, {'feature': 'region', 'sample': 'reference', 'level': 'north', 'rows': 2}, {'feature': 'region', 'sample': 'reference', 'level': 'south', 'rows': 2}]
```
<!-- /output -->

`comparison.psi` has one score per feature. Larger values mean a greater change
in fitted-level shares. `level_distribution` shows the reference/current counts
and shares behind each score. The new `"west"` category uses the fitted unseen
level; numerical bins retain the reference boundaries.

Call `analysis.compare(next_sample)` for later batches. Re-fitting on each batch
would change the reference. For temporal distributions, set `PSIConfig(period="week")` (default `"month"`)
and provide aligned
`Date`/`Datetime` timestamps with `time=` in both `fit` and `compare`.
The original ranges must satisfy `max(reference_time) < min(current_time)`;
overlap, equal endpoints, and reversed ranges raise `ValueError`. The split
may fall inside one grouping period. Comparisons without timestamps do not
have this restriction.

## Reuse feature levels

`LevelTransformer` exposes the same fit/transform operations when you need
consistent bins for another report. Continue with `reference` and `current`
above to transform both samples and compare their existing levels:

<!-- example: psi-levels -->

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

<!-- output: psi-levels -->
```text
['north', 'south', '__UNSEEN__', '__UNSEEN__']
```
<!-- /output -->

Configured columns become Polars `Categorical` columns; other columns retain
their values and order. Reuse the fitted transformer for subsequent samples.
Training's shared PSI and observed/fitted diagnostics use these levels too; see the
[shared diagnostic recipe](../configuration.md#shared-psi-and-observedfitted-diagnostics).

## API reference

Import PSI analysis from `ml_toolkit.psi`.

### Drift analysis

::: ml_toolkit.psi.PSIAnalysis

::: ml_toolkit.psi.compare_psi

### Configuration and results

::: ml_toolkit.psi.PSIConfig

::: ml_toolkit.psi.PSIResult

### Discrete-series comparison

::: ml_toolkit.psi.psi

### Reference-fitted feature levels

Import these lower-level types from `ml_toolkit.feature_levels`.
`FeatureLevelSpec` is also exported from `ml_toolkit.config` and documented in the
[configuration reference](configuration.md).

::: ml_toolkit.feature_levels.LevelTransformer

### Level definitions

::: ml_toolkit.feature_levels.LevelConfig

::: ml_toolkit.feature_levels.FeatureLevelConfig

`FeatureType` identifies numerical and categorical features. `FeatureScalar`
describes the values retained by learned levels. `NULL_LEVEL`, `ERROR_LEVEL`,
and `UNSEEN_LEVEL` are the default missing, invalid, and unseen-category labels.
