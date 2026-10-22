# PSI and feature levels

Part of [Data analysis](../data-analysis/index.md), Population Stability Index
(PSI) measures how feature distributions change
between a reference sample and a current sample. Fit categories and numerical
bins on the reference once, then reuse them for every comparison.

## Compare two samples

This small calculation makes the level counts easy to inspect; four rows per
sample are not enough to establish population drift. For monitoring real
observations, see the [recorded-hour comparison](#compare-recorded-weather-and-calendar-features).

<!-- example: psi-compare -->

```python
import polars as pl

from ml_toolkit.config import FeatureLevelConfig
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
            FeatureLevelConfig("age", "numerical", lower_bound=0),
            FeatureLevelConfig("region", "categorical"),
        ),
        n_bins=2,
    )
).fit(reference)
comparison = analysis.compare(current)
print(
    comparison.psi.sort("feature")
    .with_columns(pl.col("psi").round(4))
    .write_csv(),
    end="",
)
region_shares = (
    comparison.level_distribution.filter(
        (pl.col("feature") == "region") & (pl.col("level") != "__NULL__")
    )
    .select("level", "sample", "share")
    .pivot(on="sample", index="level", values="share")
)
print(region_shares.write_csv(), end="")
```

<!-- output: psi-compare -->
```text
feature,psi
age,6.9077
region,6.9077
level,reference,current
north,0.5,0.25
south,0.5,0.25
__UNSEEN__,0.0,0.5
```
<!-- /output -->

`comparison.psi` has one score per feature. Larger values mean a greater change
in fitted-level shares. `level_distribution` shows the reference/current counts
and shares behind each score. The new `"west"` category uses the fitted unseen
level; numerical bins retain the reference boundaries. Here north and south
each fall from 50% to 25%, and unseen categories account for 50% of current
rows. Empty levels make the score sensitive to the configured `epsilon` share
floor; this value is an arithmetic demonstration, not a universal alert threshold.

Call `analysis.compare(next_sample)` for later batches. Re-fitting on each batch
would change the reference. For temporal distributions, set `PSIConfig(period="week")` (default `"month"`)
and provide aligned
`Date`/`Datetime` timestamps with `time=` in both `fit` and `compare`.
The original ranges must satisfy `max(reference_time) <= min(current_time)`.
Equal endpoints are allowed; overlap beyond the shared endpoint and reversed
ranges raise `ValueError`. The split
may fall inside one grouping period. Comparisons without timestamps do not
have this restriction.

## Reuse feature levels

`LevelTransformer` exposes the same fit/transform operations when you need
consistent bins for another report. Continue with `reference` and `current`
above to transform both samples and compare their existing levels:

`FeatureLevelConfig` supplies the rules for each column. `fit(reference)`
determines categories or numerical intervals from the reference sample;
it does not train a prediction model. The resulting `feature_levels` tuple
contains one `FeatureLevels` per configured column, in the same order.
Each result holds its original rules in `config`, its labels in `levels`,
and its numerical `boundaries` or categorical `categories`.

<!-- example: psi-levels -->

```python
from ml_toolkit.feature_levels import LevelTransformer
from ml_toolkit.psi import compare_psi

levels = LevelTransformer(analysis.config.features, n_bins=2).fit(reference)
reference_levels = levels.transform(reference)
current_levels = levels.transform(current)
assert levels.feature_levels is not None
level_comparison = compare_psi(
    reference_levels, current_levels, feature_levels=levels.feature_levels
)
print("Age boundaries:", levels.feature_levels[0].boundaries)
print(
    "Current region levels:", current_levels["region"].cast(pl.String).to_list()
)
print("Same PSI scores:", level_comparison.psi.equals(comparison.psi))
```

<!-- output: psi-levels -->
```text
Age boundaries: (35.0,)
Current region levels: ['north', 'south', '__UNSEEN__', '__UNSEEN__']
Same PSI scores: True
```
<!-- /output -->

Configured columns become Polars `Categorical` columns; other columns retain
their values and order. Reuse the fitted transformer for subsequent samples.
Calling `fit` again replaces its levels; previously returned comparison results
retain their original immutable rules and levels.

## How PSI and observed/fitted share levels

During training, PSI and `ObservedFitted` receive the same
`ReportingConfig.features` rules and `ReportingConfig.n_bins`. Each report
creates its own `LevelTransformer` and calls `fit(train.X)` on the same training
rows for that fit. They do not share a fitted transformer, a cached level object,
or fitted state between reports.

The same rules, bin count and reference data produce identical categories and
numerical boundaries for matching features. Identical configuration alone is
insufficient when the reference data differs. `ObservedFitted(features=...)`
can select a subset of the shared feature definitions without changing how
those features are binned.

Each report applies its training-fitted levels to both train and validation/test
rows. Held-out rows do not influence the learned levels. Every enabled fit
learns levels afresh from its own training rows, so boundaries can differ across
folds and the final fit. See the
[observed/fitted guide](../evaluation/observed-fitted.md#compare-feature-levels)
for the corresponding prediction report.

## Compare recorded weather and calendar features

Prepare the [Bike Sharing cache](../evaluation/index.md#example-data). These
figures fit levels on 2011 training hours and compare 2012 test hours. Calendar
changes can be expected; PSI measures distribution differences, not their cause.
Shares count rows, independent of model weights.

<!-- example: psi-gallery-setup -->

```python
from pathlib import Path

import polars as pl

from examples.gallery_data import load_bike_pool
from ml_toolkit.training import Holdout
from ml_toolkit.config import FigureConfig

data = load_bike_pool(download_if_missing=False)
partition = Holdout(data, test_size="2012-01-01").get_test()
train, test = partition.train, partition.heldout
assert test is not None
output = Path("figures")
figure = FigureConfig(dpi=150)
print(f"2011 train hours: {len(train)}; 2012 test hours: {len(test)}")
```

<!-- output: psi-gallery-setup -->
```text
2011 train hours: 8645; 2012 test hours: 8734
```
<!-- /output -->

<!-- example: psi-gallery-scores -->

```python
from ml_toolkit.plots import psi

categorical_features = {
    "hour",
    "working_day",
    "weekday",
    "holiday",
    "season",
    "weather",
    "month",
}
weather_analysis = PSIAnalysis(
    PSIConfig(
        features=tuple(
            FeatureLevelConfig(
                name,
                "categorical" if name in categorical_features else "numerical",
            )
            for name in train.X.columns
        ),
        n_bins=10,
    )
).fit(train.X, time=train.time)
weather_comparison = weather_analysis.compare(test.X, time=test.time)
psi.feature_psi_plot(
    weather_comparison.psi, output / "psi-scores.png", figure=figure
)
print(
    weather_comparison.psi.sort("psi", descending=True)
    .head(5)
    .with_columns(pl.col("psi").round(3))
    .write_csv(),
    end="",
)
```

<!-- output: psi-gallery-scores -->
```text
feature,psi
humidity,0.048
temperature,0.036
feels_like_temperature,0.033
weather,0.004
wind_speed,0.004
```
<!-- /output -->

![Feature PSI scores comparing recorded 2011 and 2012 weather and calendar distributions.](../assets/plots/psi-scores.png)

The bars show one PSI score per feature. Inspect its level shares to understand
which values contribute to the difference; a universal alert cutoff is not
implied by this illustration.

<!-- example: psi-gallery-time -->

```python
psi.time_psi_level_distribution(
    weather_comparison.time_distribution,
    "temperature",
    output / "psi-time.png",
    figure=figure,
)
```

![Monthly temperature shares on the same 2011-fitted numerical levels.](../assets/plots/psi-time.png)

Each bar shows period-specific shares on the fitted levels. The dashed boundary
marks the first current bar. When the split falls inside a period, that period
has two equal-width bars, reference then current; the separator marks their
visual boundary, not the exact split date. Entirely missing periods stay empty.
Both plotting calls return `None`; their tables come from `PSIAnalysis`.

## Enable PSI during training

<!-- example: psi-reporting -->

```python
from ml_toolkit.config import ReportingConfig

reporting = ReportingConfig(
    psi=True, features=weather_analysis.config.features, n_bins=10
)
```

Pass this reporting config to `TrainingConfig(reporting=reporting)`. The
workflow compares train with the current held-out sample, writes `psi/scores.png`
and per-feature time distributions, and saves the score and distribution tables
as JSON. PSI uses feature values and timestamps only; no predictions are
passed to the analysis. A train-only fit has no PSI report. `features=None` infers feature
rules; `features=()` disables feature diagnostics. `epsilon` is the PSI share
floor, and `period` sets temporal grouping. Scheduling and shared figure settings
are described in the [data analysis overview](../data-analysis/index.md).

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
`FeatureLevelConfig` is also exported from `ml_toolkit.config` and documented in the
[configuration reference](configuration.md).

::: ml_toolkit.feature_levels.LevelTransformer

### Level definitions

::: ml_toolkit.feature_levels.FeatureLevels

`FeatureType` identifies numerical and categorical features. `FeatureScalar`
describes the values retained by learned levels. `NULL_LEVEL`, `ERROR_LEVEL`,
and `UNSEEN_LEVEL` are the default missing, invalid, and unseen-category labels.

### PSI plots

::: ml_toolkit.plots.psi
    options:
      heading_level: 3
      members: [feature_psi_plot, time_psi_level_distribution]
