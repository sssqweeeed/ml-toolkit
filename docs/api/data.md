# Data

Use a `Pool` to keep model inputs and row metadata aligned. `X` contains only
the features the model should see; targets, dates, weights, and optional IDs
travel alongside it.

## Prepare a pool

This small dataset is also used in the [training](training.md) and
[model](models.md) examples. You can replace `frame` with your own Polars data.

```python
from datetime import date, timedelta

import polars as pl

from ml_toolkit import Pool

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

There are 24 observations and two model features. Each timestamp is a Polars
`Date`; the weight of `1.0` gives every observation equal influence.
Training requires ascending, non-null dates, finite targets, and finite,
nonnegative weights with positive total in each split.

## Slice without losing alignment

```python
first_week = data.take(range(7))
age_only = first_week.select_features(["age"])

print(len(age_only))          # 7
print(age_only.X.columns)     # ['age']
print(len(data))              # 24
```

Both operations return new pools and carry along targets, dates, weights, and
IDs. `to_model_input()` produces a pandas feature frame for a backend or serving
example, converting categorical columns to strings and filling missing
categories with `"unknown"`.

## Array types

Numerical APIs use `FloatArray` for NumPy arrays with `float64` values and
`IntArray` for arrays with `int64` values. Import either from `ml_toolkit.types`
when annotating your own functions. Each API documents the expected shape;
targets, predictions, and row weights normally contain one value per row.
The shared `ModelParams` alias is a dictionary of backend parameter names to
values, as shown in the [model example](models.md#fit-and-predict-directly).

## API reference

Projects can also supply an object implementing the `TrainingData` protocol.

::: ml_toolkit.data.Pool
    options:
      members: [take, select_features, with_target, to_model_input, to_catboost]

::: ml_toolkit.training.TrainingData
