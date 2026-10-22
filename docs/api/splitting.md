# Data splitting

Bind a prepared `Pool` to a splitter, then pass that object to `train_model`.
The same object also supplies pairs for direct fitting outside the workflow.

## Choose a strategy

| Strategy | Validation | Main training |
| --- | --- | --- |
| `Holdout` | One trailing validation sample | All rows before optional test |
| `TimeSeriesCV` | Expanding or bounded chronological windows | All rows before optional test |
| Custom `Splitter` | Pairs yielded by your implementation | Exactly `get_test().train` |

Input pools require ascending timestamps and aligned finite targets and weights,
with nonnegative weights and positive total weight. Splits preserve metadata,
row order and the concrete pool type. See [Data](data.md) for preparing a pool.

## Inspect chronological windows

Run from an installed checkout after `python -m examples.prepare_data --dataset bike`.
The imported `prepare_training()` loads the original Bike Sharing hours, model
parameters and local MLflow settings described in the [training example](training.md#run-a-holdout-experiment).
Targets are rentals per hour; weights are one per recorded hour.

<!-- example: splitting-setup -->

```python
from examples.training_setup import prepare_training
from ml_toolkit import train_model
from ml_toolkit.metrics import metrics_frequency
from ml_toolkit.models import CatBoostLogLinkRegressionModel
from ml_toolkit.training import Holdout

data, params, config = prepare_training()
splitter = Holdout(data, val_size=0.25, test_size=0.2)
```

<!-- example: splitting-windows -->

```python
import pandas as pd
from ml_toolkit.data import Pool
from ml_toolkit.training import Splitter, TimeSeriesCV


def split_table(strategy: Splitter[Pool]) -> pd.DataFrame:
    rows = []
    for part in strategy.get_validation():
        assert part.heldout is not None
        for sample, pool in (("train", part.train), ("val", part.heldout)):
            rows.append(
                (
                    str(part.index),
                    sample,
                    len(pool),
                    str(pool.time[0])[:10],
                    str(pool.time[-1])[:10],
                )
            )
    final = strategy.get_test()
    for sample, pool in (("train", final.train), ("test", final.heldout)):
        if pool is not None:
            rows.append(
                (
                    "final",
                    sample,
                    len(pool),
                    str(pool.time[0])[:10],
                    str(pool.time[-1])[:10],
                )
            )
    return pd.DataFrame(
        rows, columns=["fold", "sample", "rows", "first date", "last date"]
    )


print("Holdout")
print(split_table(splitter).to_string(index=False))
cv = TimeSeriesCV(data, n_splits=3, test_size=0.2, gap=24)
print("\nTimeSeriesCV")
print(split_table(cv).to_string(index=False))
```

<!-- output: splitting-windows -->
```text
Holdout
 fold sample  rows first date  last date
    1  train 10427 2011-01-01 2012-03-15
    1    val  3476 2012-03-15 2012-08-07
final  train 13903 2011-01-01 2012-08-07
final   test  3476 2012-08-07 2012-12-31

TimeSeriesCV
 fold sample  rows first date  last date
    1  train  3454 2011-01-01 2011-05-28
    1    val  3475 2011-05-29 2011-10-22
    2  train  6929 2011-01-01 2011-10-21
    2    val  3475 2011-10-22 2012-03-15
    3  train 10404 2011-01-01 2012-03-14
    3    val  3475 2012-03-15 2012-08-07
final  train 13903 2011-01-01 2012-08-07
final   test  3476 2012-08-07 2012-12-31
```
<!-- /output -->

The tables show calendar dates for compact display. Counts use the exact
hourly observations; adjacent samples can begin and end on the same date
without sharing rows.

Each CV fold leaves 24 recorded rows between training and validation. They are
included again in the main train. Rows need not be equally spaced in clock
time: integer `gap`, integer sizes and `max_train_size` count observations.

`test_size` accepts a row count, fraction of the original pool, calendar date,
`YYYY-MM-DD` string, or `None`. Dates start test inclusively in the data's timezone;
when a date has no observations, the next observation starts test. Datetimes are
not accepted. Fractions round upward, and both resulting samples must be nonempty.
Validation fractions use only the retained non-test rows.

`TimeSeriesCV.val_size=None` divides the retained rows into equal validation
blocks. `max_train_size` limits each CV training window, but not final training;
`skip_splits` omits initial folds and preserves their original one-based indices.
Without test, built-in final training uses the entire pool.

## Use a duration gap

Pass `datetime.timedelta` directly; no conversion to milliseconds is needed.

<!-- example: splitting-duration-gap -->

```python
from datetime import timedelta

duration_cv = TimeSeriesCV(data, n_splits=3, test_size=0.2, gap=timedelta(days=1))
for part in duration_cv.get_validation():
    assert part.heldout is not None
    assert part.train.time[-1] <= part.heldout.time[0] - timedelta(days=1)
```

`gap=timedelta(seconds=1)` is also supported. Preceding rows are retained when
`time <= validation_start - gap`, including equality. Zero duration matches
zero row gap. The duration measures elapsed time across timezone/DST changes;
`Date` values represent midnight and sub-day gaps are not rounded. A duration
of 30 days is exactly 30 days, not a calendar month.

Validation windows and the test boundary stay unchanged. CV applies
`max_train_size` after excluding the gap. Negative gaps or an empty train in
a used fold raise `ValueError`. Gap rows remain available for later fitting.
`parameters()` stores row gaps as integers and durations as
`{"type": "timedelta", "microseconds": N}`.

## Fit outside the workflow

Continue with `splitter`, `params` and the model imported above:

<!-- example: splitting-direct -->

```python
part = next(splitter.get_validation())
assert part.heldout is not None
model = CatBoostLogLinkRegressionModel(params).fit(part.train, part.heldout)
final = splitter.get_test()
final_model = CatBoostLogLinkRegressionModel(model.get_refit_params()).fit(
    final.train
)
assert final.heldout is not None
prediction = final_model.predict_pool(final.heldout)
print("Final test predictions:", prediction.shape)
```

<!-- output: splitting-direct -->
```text
Final test predictions: (3476,)
```
<!-- /output -->

Each `get_validation()` call creates a fresh iterator. Direct use leaves
iteration, metrics and logging to the caller. `train_model` consumes one pair
at a time and calls `get_test()` once after validation, so test pools are not
retained during validation. With `fit_on_test=True`,
it calls `get_final()` after test evaluation for an additional full-data fit.

## Implement a splitter

Subclass `Splitter` and implement two methods. `get_validation()` yields
`Split(train, heldout, index)` with a nonempty validation `heldout` and positive,
unique indices. Iterator order determines execution order. At least one pair
is required. `get_test()` returns `Split(train, heldout)` where `heldout` is
optional test; its index is ignored. Its train contains every eligible non-test
row, or the entire eligible pool when test is absent. The separate `get_final()`
method defaults to `self.data`; override it to customize the full pool used for
additional training including test. It returns a pool, not a `Split`.

This reusable example uses explicit positional boundaries:

??? note "Complete implementation"

    ```python
    --8<-- "examples/custom_splitter.py"
    ```

Use it in the ordinary workflow:

<!-- example: splitting-custom -->

```python
from examples.custom_splitter import ProjectSplitter

project_splitter = ProjectSplitter(
    data, validation_start=len(data) // 2, test_start=3 * len(data) // 4
)
project_result = train_model(
    splitter=project_splitter,
    params=params,
    model_factory=CatBoostLogLinkRegressionModel,
    metric=metrics_frequency,
    run_name="project-splits",
    config=config,
)
print("ProjectSplitter")
print(split_table(project_splitter).to_string(index=False))
print("Validation folds fitted:", len(project_result.validation.splits))
```

<!-- output: splitting-custom -->
```text
ProjectSplitter
 fold sample  rows first date  last date
    1  train  8689 2011-01-01 2012-01-02
    1    val  4345 2012-01-02 2012-07-02
final  train 13034 2011-01-01 2012-07-02
final   test  4345 2012-07-02 2012-12-31
Validation folds fitted: 1
```
<!-- /output -->

Keep the original pool in `self.data`; generate a fresh iterator on each call.
The workflow checks valid pools, matching feature schemas and declarations within
each pair, and selected-feature availability in final data. Your splitter owns
sample membership and leakage prevention. IDs are optional and may repeat; no
ID-based overlap check is performed. Preserve row metadata by using `Pool.take`.

`parameters()` defaults to `{}`. Override it with explicit JSON-compatible
settings to log them alongside the class name in `configuration/splitter.json`.
Do not return data, generators or fitted models. The splitter supplies data;
model fitting, refitting and serving belong to the [model adapter](models.md).

## API reference

::: ml_toolkit.training.Splitter
    options:
      merge_init_into_class: true
      members: [get_validation, get_test, get_final, parameters]

::: ml_toolkit.training.Split
    options:
      members: [train, heldout, index]

::: ml_toolkit.training.Holdout
    options:
      merge_init_into_class: true
      members: [get_validation, get_test, get_final, parameters]

::: ml_toolkit.training.TimeSeriesCV
    options:
      merge_init_into_class: true
      members: [get_validation, get_test, get_final, parameters]
