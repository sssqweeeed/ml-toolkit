# Data analysis

Analyze input data independently of model predictions. Use `data_analyzers`
for distributions and custom data reports, and `prediction_evaluators` for
reports that compare or explain model predictions. **Data analysis** covers
exploratory data analysis (EDA) as well as comparisons and drift monitoring.

| Question | Guide | Interface |
| --- | --- | --- |
| How does a numerical feature change over time? | [Numerical distributions](distributions.md) | `NumericalFeatureDistributions` |
| How have feature shares changed? | [PSI and feature levels](../api/psi.md) | `PSIAnalysis` directly; `ReportingConfig(psi=True)` in training |
| How do I produce project-specific data metrics or files? | [Build a data analyzer](custom.md) | `DataAnalyzer` |
| How accurate or well calibrated are predictions? | [Evaluation](../evaluation/index.md) | `PredictionEvaluator` |

Observed/fitted reports belong to Evaluation even when grouped by feature:
their quantities depend on predictions. Feature importance and SHAP explain a
fitted model and also remain there. Feature levels are shared infrastructure
for both observed/fitted reports and PSI; their API stays in
`ml_toolkit.feature_levels`. Both reports share configuration and training rows,
and independently fit their own transformers. The
[feature-level contract](../api/psi.md#how-psi-and-observedfitted-share-levels)
explains how their levels remain consistent.

## Add reports to training

This function accepts a prepared `Pool` and model parameters. Set your MLflow
tracking URI and experiment before calling it. The two report lists make their
responsibilities explicit:

<!-- example: data-analysis-workflow -->

```python
from ml_toolkit import Pool, train_model
from ml_toolkit.config import ReportingConfig, TrackingConfig, TrainingConfig
from ml_toolkit.data_analysis import NumericalFeatureDistributions
from ml_toolkit.evaluation import ObservedFitted
from ml_toolkit.models import CatBoostRegressionModel
from ml_toolkit.training import Holdout
from ml_toolkit.types import ModelParams
from examples.custom_metric import weighted_mae


def train_with_reports(data: Pool, params: ModelParams):
    return train_model(
        splitter=Holdout(data, test_size=0.2),
        params=params,
        model_factory=CatBoostRegressionModel,
        metric=weighted_mae,
        run_name="data-and-predictions",
        config=TrainingConfig(
            reporting=ReportingConfig(psi=True),
            tracking=TrackingConfig(log_final_model=False),
        ),
        prediction_evaluators=[ObservedFitted()],
        data_analyzers=[NumericalFeatureDistributions(period="month")],
    )
```

PSI keeps its existing configuration and artifact paths. The distribution
report still writes `numerical_feature_distributions/<index>_<feature>.png`.
No statistical calculations or plot formats change in this separation.

## Inputs, outputs and scheduling

Pass one `DataAnalyzer`, a list, `None`, or `[]`. Each hook receives an aligned
`Pool` and non-null String `split` labels: `train` plus at most one of `val`
or `test`. There is no `prediction` or model argument. The existing `Pool`
container still includes targets, weights and timestamps; no new data container
is needed. Direct calls can analyze prepared data before fitting any model.

During training, analyzers run after prediction evaluators on the same selected
features and combined rows. They use train + val for tuning and validation,
and train + test (or train only) for final fitting. `ReportingConfig.on_tuning`,
`.on_validation` and `.on_final` gate both interfaces and PSI/SHAP. Their defaults
are false, true and true. These switches do not change core training metrics.

Like evaluators, analyzers may return finite scalar metrics from `evaluate()`
and selected files from `write_artifacts()`. Either hook can be omitted.
Metrics use `<name>/<key>`; artifacts use `<name>/`. They do not enter CV
averages, training results or optimization objectives. Instances are reused
across fits and must not retain data or fitted state or mutate inputs.

Names must be unique ignoring case across **both** report lists. Enabled PSI
and SHAP reserve `psi` and `shap`. Names start with an ASCII letter or digit
and contain only ASCII letters, digits, underscores, hyphens or dots.
Only returned files inside `artifact_dir` are uploaded; temporary directories
are removed on success and failure. The [shared artifact contract](../evaluation/index.md#reporting-inputs-and-outputs)
and [figure/time settings](../evaluation/index.md#figure-settings-and-time) apply.
Errors identify the analyzer and training stage.

Settings are recorded separately in `configuration/data_analyzers.json` and
`configuration/evaluators.json`; each contains names, classes and explicit
parameters. `None` or `[]` disables custom analyzers, while PSI remains governed
by `ReportingConfig.psi`.

## Update existing distribution reports

Import `NumericalFeatureDistributions` from `ml_toolkit.data_analysis`, move the
instance from `prediction_evaluators` to `data_analyzers`, and omit `prediction`
in direct hook calls. Keep your existing settings and artifact paths.

The old `ml_toolkit.evaluation.NumericalFeatureDistributions` import remains a
compatibility adapter with its previous prediction-based signature. Existing
`PredictionEvaluator` subclasses and other built-in evaluators remain usable
without changes. New data reports should implement `DataAnalyzer`.

## Example data

Follow the [source data preparation](../evaluation/index.md#example-data) once.
The distribution and PSI guides load the cached observations directly and
compare 2011 with 2012 without fitting a prediction model.
