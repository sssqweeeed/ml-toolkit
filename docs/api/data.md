# Data

`Pool` is the interface between your project's data container and the toolkit.
It describes aligned Polars features, targets, dates, weights, and row operations.
The package provides the protocol, while your project supplies the implementation.
Do not instantiate `Pool` itself.

## Prepare a pool

The repository includes `examples/project_pool.py`, a concrete `ProjectPool`
implementation for the examples. From the cloned repository, run:

<!-- example: data-prepare -->

```python
from datetime import date, timedelta

import polars as pl

from examples.project_pool import ProjectPool

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
data = ProjectPool(
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
model features. `ProjectPool` is example code outside the installed package;
copy it into your project or implement the interface on your existing container.

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
produces a pandas feature frame; the example implementation converts categorical
columns to strings and fills missing categories with `"unknown"`.

Training reserves the literal category `"unknown"` for missing values. Raw
categorical columns in training and validation inputs must not already contain
that value after string conversion, even when they have no missing values.
Training reports the offending column. Prepared model inputs, prediction and
serving continue to accept `"unknown"`; ordinary missing values are still filled.

## Implement your own container

Import the protocol for annotations or static checking; explicit inheritance
is unnecessary:

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
and finite nonnegative weights with a positive total. Optional attributes must
exist even when their value is `None`. Read-only properties or ordinary dataclass
fields can satisfy the protocol.

### Row and feature operations

`take` preserves requested row order, including repeated indices.
`select_features` preserves all row metadata and filters feature declarations.
`with_target` preserves rows and their metadata. Each method returns the same
concrete container type, so chronological splitting and selection retain your
project's data representation.

### Backend conversion

The bundled CatBoost adapters call `to_catboost` for fitting and
`to_model_input` for named feature frames and serving examples. Preserve fitted
feature names and categorical handling in both conversions. A scalar baseline
must be broadcast across rows; array baselines must stay aligned.

??? example "Complete project-owned implementation"

    This is the runnable source used by the examples and tests. It is not
    installed as part of `ml_toolkit`.

    ```python
    --8<-- "examples/project_pool.py"
    ```

## Array types

Import `FloatArray` and `IntArray` from `ml_toolkit.types` when annotating your
functions. They mean NumPy `float64` and `int64` arrays; individual APIs describe
the required shape. `ModelParams` is the backend parameter dictionary.

## API reference

### Data interface

::: ml_toolkit.Pool
    options:
      members: false

### Row and feature operations

::: ml_toolkit.Pool.take

::: ml_toolkit.Pool.select_features

::: ml_toolkit.Pool.with_target

### Backend conversion

::: ml_toolkit.Pool.to_model_input

::: ml_toolkit.Pool.to_catboost
