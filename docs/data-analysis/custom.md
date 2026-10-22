# Build a data analyzer

Implement `DataAnalyzer` when project-specific metrics or files depend on input
data, not predictions. Use [PredictionEvaluator](../evaluation/custom.md) when
predictions are part of the calculation. Both interfaces support optional
`evaluate()`, `write_artifacts()` and `parameters()` hooks.

## Implement the contract

This example counts rows by sample and writes a per-feature null-count table.
It is an example extension, not an additional built-in analyzer:

```python
--8<-- "examples/custom_data_analyzer.py"
```

The method signatures contain no model or prediction input. Keep instances
stateless; override `parameters()` if your analyzer has settings to record.
The default is `{}`. Return explicit JSON-compatible values, not fitted state.

## Use it without fitting a model

Prepare the [cached observations](index.md#example-data), then run:

<!-- example: custom-data-analyzer-direct -->

```python
from pathlib import Path

import polars as pl

from examples.custom_data_analyzer import DataAudit
from examples.gallery_data import load_bike_pool
from ml_toolkit.config import FigureConfig
from ml_toolkit.training import Holdout

data = load_bike_pool(download_if_missing=False)
partition = Holdout(data, test_size="2012-01-01").get_test()
assert partition.heldout is not None
split = pl.Series(
    "split",
    ["train"] * len(partition.train) + ["test"] * len(partition.heldout),
)
analyzer = DataAudit(name="data_audit")
print(analyzer.evaluate(data=data, split=split))
files = analyzer.write_artifacts(
    data=data,
    split=split,
    artifact_dir=Path("data-audit"),
    figure=FigureConfig(),
    features=None,
    n_bins=10,
)
print([path.as_posix() for path in files])
```

<!-- output: custom-data-analyzer-direct -->
```text
{'train/rows': 8645.0, 'test/rows': 8734.0}
['null_counts.csv']
```
<!-- /output -->

Direct calls return values and file paths; they do not log to MLflow.
To use the same instance during training, pass `data_analyzers=analyzer` or
`data_analyzers=[analyzer]` to `train_model`. Training logs
`data_audit/train/rows`, `data_audit/val/rows` or `data_audit/test/rows`, and
`data_audit/null_counts.csv`. Train-only calls still produce a row count and
table in this example; each analyzer decides whether it needs a comparison pair.

The [reporting contract](index.md#inputs-outputs-and-scheduling) covers names,
stage switches, shared settings, metric validation and artifact cleanup.

## API reference

::: ml_toolkit.data_analysis.DataAnalyzer
    options:
      heading_level: 3
      members: [evaluate, write_artifacts, parameters]
