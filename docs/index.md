# ml-toolkit

Prepare data, choose a model and metric, then train, evaluate and record an
experiment with the same workflow across tasks. Splitter and evaluator objects
control data partitions and additional reports.

## Start here

- [Getting started](getting-started.md): installation and one complete training run.
- [Task recipes](configuration.md): model, metric and report choices for your target.
- [Data](api/data.md): prepare a `Pool` and preserve row alignment when slicing.
- [Training](api/training.md): run the workflow and interpret its result.
- [Models](api/models.md): fit, predict, save and load a supported adapter.

## Follow the workflow

1. A splitter supplies train/validation pairs.
2. Each pair independently selects features, tunes parameters and fits a model.
   Selection and tuning are optional.
3. A fresh final model uses the last pair's selected settings and the training
   pool returned by `splitter.get_final()`.
4. An optional final held-out pool is evaluated after fitting. Metrics, reports
   and model artifacts are recorded in MLflow.

Built-in splitters reserve chronological windows and refit on all non-test rows.
[Custom splitters](api/splitting.md) define their own windows and final training data.
The caller prepares features and targets and selects the MLflow experiment.

## Inspect and customize

| Task | Guide |
| --- | --- |
| Choose a diagnostic and understand its graph | [Evaluation](evaluation/index.md) |
| Analyze input distributions or drift | [Data analysis](data-analysis/index.md) |
| Supply custom data metrics or artifacts | [Build a data analyzer](data-analysis/custom.md) |
| Supply extra prediction metrics or artifacts | [Build an evaluator](evaluation/custom.md) |
| Define validation windows | [Data splitting](api/splitting.md) |
| Select or rank model features | [Feature selection](api/feature-selection.md) |
| Search model parameters | [Parameter tuning](api/tuning.md) |
| Implement model fitting and serving | [Build a model adapter](api/custom-model.md) |
| Choose or implement a training metric | [Metrics](api/metrics.md) |
| Find experiment results | [Tracking](api/tracking.md) |
| Configure shared settings | [Configuration](api/configuration.md) |

Each guide includes executable usage and the relevant API reference. The site
follows the `main` branch; source and examples are in the
[GitHub repository](https://github.com/sssqweeeed/ml-toolkit).
