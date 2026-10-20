# Data

`Pool` is the toolkit's ready-to-use data container for aligned Polars features,
targets, dates, weights, and row metadata. Instantiate it with your prepared
data; row operations and backend conversions are included in the package.

## Prepare a pool

After installing the package, create a pool directly:

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

print(len(age_only))
print(age_only.X.columns)
print(len(data))
```

<!-- output: data-slice -->
```text
7
['age']
24
```
<!-- /output -->

These operations return new containers and carry targets, dates, weights, IDs,
and groups together. `with_target` replaces only the target. `to_model_input()`
produces a pandas feature frame; the container converts categorical
columns to strings and fills missing categories with `"unknown"`.

Training reserves the literal category `"unknown"` for missing values. Raw
categorical columns in training and validation inputs must not already contain
that value after string conversion, even when they have no missing values.
Training reports the offending column. Prepared model inputs, prediction and
serving continue to accept `"unknown"`; ordinary missing values are still filled.

## Use the shared container

The same class is available from `ml_toolkit` and `ml_toolkit.data` for
construction and type annotations:

<!-- example: data-protocol -->

```python
from ml_toolkit import Pool

rows: Pool = data
print(len(rows))
```

<!-- output: data-protocol -->
```text
24
```
<!-- /output -->

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
