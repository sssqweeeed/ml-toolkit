# Data

`Pool` is the toolkit's ready-to-use data container for aligned Polars features,
targets, dates, weights, and row metadata. Instantiate it with your prepared
data; row operations and backend conversions are included in the package.

## Prepare a pool

After installing the package, create a pool directly. This small constructed
fixture demonstrates container operations; it is not an evaluation dataset:

<!-- example: data-prepare -->

```python
from datetime import date, timedelta

import polars as pl

from ml_toolkit.data import Pool

n_rows = 24
frame = pl.DataFrame(
    {
        "age": [20 + i % 10 for i in range(n_rows)],
        "region": ["north", "south"] * (n_rows // 2),
        "target": [float(i % 3) for i in range(n_rows)],
        "date": [date(2025, 1, 1) + timedelta(days=i) for i in range(n_rows)],
        "weight": [1.0] * n_rows,
    }
).sort("date")
data = Pool(
    X=frame.select("age", "region"),
    y=frame["target"],
    time=frame["date"],
    weight=frame["weight"],
    cat_features=("region",),
    numerical_features=("age",),
)
```

There are 24 rows and two model features. This `data` is reused by the
[training](training.md) and [model](models.md) examples. Targets, timestamps,
weights, IDs, and reporting groups stay outside `X` unless they are intentional
model features. No project-specific container implementation is required.

## Slice without losing alignment

<!-- example: data-slice -->

```python
first_week = data.take(range(7))
age_only = first_week.select_features(["age"])

print("Selected rows and features:", len(age_only), age_only.X.columns)
reordered = data.take([2, 0, 2])
print("Aligned (age, target, date):")
for age, target, time in zip(reordered.X["age"], reordered.y, reordered.time):
    print(age, target, time)
print("Original rows:", len(data))
```

<!-- output: data-slice -->
```text
Selected rows and features: 7 ['age']
Aligned (age, target, date):
22 2.0 2025-01-03
20 0.0 2025-01-01
22 2.0 2025-01-03
Original rows: 24
```
<!-- /output -->

The selection repeats row 2 around row 0; their ages, targets, and dates move
together. Weights and optional IDs and reporting groups follow the same order.
Feature selection filters feature declarations as well as `X`. Both operations
return new containers, leaving the original 24 rows intact. Reordered data is
useful for inspection; training still requires ascending timestamps.

## Replace a target without changing rows

For example, prepare a log-transformed target while keeping original counts
available for comparison:

<!-- example: data-target -->

```python
transformed = data.with_target(data.y.cast(pl.Float64).log1p())
print("Original targets:", data.y.head(3).to_list())
print("Log1p targets:", transformed.y.head(3).round(4).to_list())
print("Dates preserved:", transformed.time.equals(data.time))
```

<!-- output: data-target -->
```text
Original targets: [0.0, 1.0, 2.0]
Log1p targets: [0.0, 0.6931, 1.0986]
Dates preserved: True
```
<!-- /output -->

A model fitted to this transformed target predicts on the transformed scale;
convert predictions back before comparing them with original counts. The
original pool remains available with unchanged targets.

## Convert features and baseline for a backend

This conversion example includes a missing category. Use a small copy of the
prepared pool so the shared `data` keeps its original values:

<!-- example: data-conversion -->

```python
from dataclasses import replace

with_missing = replace(
    data.take([0, 1, 2]),
    X=data.X.head(3).with_columns(pl.Series("region", ["north", None, "south"])),
)
model_input = with_missing.to_model_input()
print("Backend categories:", model_input["region"].tolist())
print("Source categories:", with_missing.X["region"].to_list())
native = with_missing.to_catboost(baseline=0.25)
print("Backend features:", native.get_feature_names())
print("Raw-score baseline:", native.get_baseline().reshape(-1).tolist())
```

<!-- output: data-conversion -->
```text
Backend categories: ['north', 'unknown', 'south']
Source categories: ['north', None, 'south']
Backend features: ['age', 'region']
Raw-score baseline: [0.25, 0.25, 0.25]
```
<!-- /output -->

`to_model_input()` returns pandas feature columns, converts categorical values
to strings, and fills missing values with `"unknown"`. `to_catboost()` also
carries targets, weights, and timestamps, broadcasting a scalar baseline to
every row. The baseline is a raw model score, not a prediction on the response
scale; a log-link model interprets 0.25 on the log scale.

Training reserves the literal category `"unknown"` for missing values. Raw
categorical columns in training and validation inputs must not already contain
that value after string conversion, even when they have no missing values.
Training reports the offending column. Prepared model inputs, prediction and
serving continue to accept `"unknown"`; ordinary missing values are still filled.

## Use the shared container

The same class is available from `ml_toolkit` and `ml_toolkit.data` for
construction and type annotations:

<!-- example: data-import-alias -->

```python
from ml_toolkit import Pool
```

### Row fields

| Fields | Contract |
| --- | --- |
| `X`, `y`, `time`, `weight` | Aligned, nonempty Polars frame/series; one value per row |
| `cat_features`, `numerical_features` | Tuples of feature names present in `X` |
| `object_id` | Aligned string/integer IDs or `None`; required for custom evaluation |
| `business_type` | Aligned diagnostic groups or `None` |

Training validates ascending non-null Date/Datetime timestamps, finite targets,
and finite nonnegative weights with a positive total. Optional metadata defaults
to `None`. Fields cannot be reassigned; contained frames and series are not
deeply immutable.

### Row and feature operations

`take` preserves requested row order, including repeated indices.
`select_features` preserves all row metadata and filters feature declarations.
`with_target` preserves rows and their metadata. Each method returns the same
concrete container type, so chronological splitting and selection retain your
container type.

### Backend conversion

The bundled CatBoost adapters call `to_catboost` for fitting and
`to_model_input` for named feature frames and serving examples. Preserve fitted
feature names and categorical handling in both conversions. A scalar baseline
must be broadcast across rows; array baselines must stay aligned.

## Array types

Import `FloatArray` and `IntArray` from `ml_toolkit.types` when annotating your
functions. They mean NumPy `float64` and `int64` arrays; individual APIs describe
the required shape. `ModelParams` is the backend parameter dictionary.

## API reference

### Data container

::: ml_toolkit.data.Pool
    options:
      members: false

### Row and feature operations

::: ml_toolkit.data.Pool.take

::: ml_toolkit.data.Pool.select_features

::: ml_toolkit.data.Pool.with_target

### Backend conversion

::: ml_toolkit.data.Pool.to_model_input

::: ml_toolkit.data.Pool.to_catboost
