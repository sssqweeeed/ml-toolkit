# Evaluation

Use `prediction_evaluators` to add reports to `train_model`, or call plotting
functions directly with predictions you already have. Choose a report by the
question you want to answer. For feature distributions, drift, or custom
reports that do not use predictions, see [Data analysis](../data-analysis/index.md):

| Question | Guide | Reports |
| --- | --- | --- |
| Which probability threshold balances precision and recall? Are probabilities calibrated? | [Classification](classification.md) | `PrecisionRecall`, `F1Threshold`, `Calibration` |
| Where do predicted means differ from observations? | [Observed versus fitted](observed-fitted.md) | `ObservedFitted`, `TimeObservedFitted` |
| How much outcome is captured in the highest-ranked rows? | [Ranking](ranking.md) | `Gains`, `Lift` |
| How are known positives ranked among unlabeled rows? | [Positive-unlabeled](pu.md) | `PUScoreDistribution`, `PUPositiveCapture` |
| Which features contribute to predictions? | [Feature importance and SHAP](explainability.md) | `ReportingConfig(shap=ShapConfig(...))` and direct plots |

Every family guide places evaluator configuration beside the resulting figures,
interpretation, and direct plotting functions. [Build an evaluator](custom.md)
when you need an additional metric or artifact.

Install example dependencies from a source checkout and explicitly cache the
original sources before running recipes:

```bash
python -m pip install -r requirements-examples.txt
python -m examples.prepare_data
```

## Add reports to training

Pass one evaluator, a list, `None`, or `[]`. This example uses hourly bike rental
counts and records feature-level and monthly comparisons in each enabled run.

<!-- example: evaluation-workflow -->

```python
from pathlib import Path

import mlflow

from examples.gallery_data import load_bike_pool
from ml_toolkit import train_model
from ml_toolkit.config import ReportingConfig, TrackingConfig, TrainingConfig
from ml_toolkit.evaluation import ObservedFitted, TimeObservedFitted
from examples.custom_metric import weighted_mae
from ml_toolkit.models import CatBoostRegressionModel
from ml_toolkit.training import Holdout

mlflow.set_tracking_uri(f"sqlite:///{Path('mlflow.db').resolve()}")
mlflow.set_experiment("evaluation-examples")
data = load_bike_pool(download_if_missing=False)
reports = [ObservedFitted(features=("temperature",)), TimeObservedFitted()]
result = train_model(
    splitter=Holdout(data, test_size="2012-01-01"),
    params={
        "iterations": 80,
        "depth": 5,
        "random_seed": 42,
        "thread_count": 2,
        "verbose": False,
        "allow_writing_files": False,
    },
    model_factory=CatBoostRegressionModel,
    metric=weighted_mae,
    run_name="bike-evaluation",
    config=TrainingConfig(
        reporting=ReportingConfig(on_validation=False, on_test=True),
        tracking=TrackingConfig(log_test_model=False, log_final_model=False),
    ),
    prediction_evaluators=reports,
)
print("Test MAE (rentals/hour):", round(result.test_metrics["mae"], 2))
```

<!-- output: evaluation-workflow -->
```text
Test MAE (rentals/hour): 91.4
```
<!-- /output -->


The [observed/fitted guide](observed-fitted.md) shows these two report types and
how to inspect their aggregate tables. It uses a fixed direct fit to make the
illustrations independently reproducible.

## Reporting inputs and outputs

Each call receives one aligned `Pool`, one finite prediction per row, and a
non-null String `split` Series containing `train` and at most one of `val` or
`test`. The split labels are separate from model features; object IDs are
optional. Training passes train + val during validation and tuning, and train +
test during the `test` stage. The `final_fit` stage receives only train rows.

Classification, ranking, and PU reports use only the held-out rows. Comparison
reports show train and held-out rows together. Builtins skip incomplete pairs
or absent held-out rows. Builtin `evaluate()` methods return `{}`; their output
is graphical. Prediction evaluators use the supplied predictions; SHAP uses
the fitted native model.

`ReportingConfig.on_tuning`, `.on_validation`, `.on_test`, and `.on_final_fit`
control both methods of every evaluator and data analyzer, plus PSI/SHAP. Defaults are false, true,
true, and true. Reports that require held-out rows skip `final_fit`.
Core training metrics are calculated independently. PSI and SHAP are opt-in.
Shared `figure`, feature-level definitions, and `n_bins` come from that config;
each report page explains which settings it uses.

Names must be unique across evaluators and data analyzers ignoring case, start with an ASCII letter or digit, and
contain only ASCII letters, digits, underscores, hyphens, or dots. Enabled PSI
and SHAP reserve `psi` and `shap`. Give repeated report instances distinct names.
Extra metrics are logged as `<name>/<key>`, and do not enter CV averages,
training results, or optimization objectives.

Only returned files are uploaded beneath `<name>/`. Single-sample reports use
`val/` or `test/` subdirectories; comparison reports use their report root.
The family pages show exact filenames. Temporary directories are removed on
success and failure. Evaluators may return PNG, SVG, PDF, CSV, Parquet, JSON,
or HTML files. Errors include evaluator and training-stage context.

## Figure settings and time

`FigureConfig` sets dimensions, DPI and `format="png"`, `"svg"`, or `"pdf"`.
Saving functions create parent directories, write the figure, and close it.
Direct functions return their documented tables or `None`; `binary_curves`
calculates a table without saving a figure.

Temporal reports accept `minute`, `hour`, `day`, `week`, `month`, `quarter`, and
`year`; the default is `month`. Grouping uses the input timezone, which must
match across the pair. Weeks begin on Monday and quarters are calendar
quarters. Minute/hour require `Datetime`. `period_start` is the beginning of
the calendar period in the input timezone.

Original timestamps must satisfy `max(train_time) <= min(heldout_time)`.
Equal endpoints are allowed. Splits inside a calendar period are supported;
line plots retain two sample points at that period start. Missing periods have
no computed markers, and lines connect neighboring observed periods.

## Example data

Most figures use the [UCI Bike Sharing dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)
(Hadi Fanaee-T, CC BY 4.0): 17,379 recorded hourly observations. The model fits
2011, and all 2012 hours are held out. Dates combine the recorded date and hour.
Targets are hourly rental counts; each hour has weight one. The eleven weather
and calendar predictors exclude `casual` and `registered`, whose sum reveals
the target. Temperature and humidity retain the source's normalized scales.
Classification defines a positive hour as at least 200 rentals before fitting.
The [PU guide](pu.md) uses recorded Cora citations and unverified candidate links.

The reusable loader owns the exact model parameters and chronological split;
it never downloads missing data during recipe execution.

??? example "Inspect data preparation and model fitting"

    ```python
    --8<-- "examples/evaluation_data.py"
    ```

Regenerate all figures and captured outputs with
`python -m examples.render_plots`; add `--check` to detect stale outputs or
images without rewriting documentation. The broader
`python -m examples.render_examples --check` also checks the remaining guides.
