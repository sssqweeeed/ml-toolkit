# Migrating to 7.0

Version 7.0 reorganizes the package so that every concept has one home. Only
import paths change in this part of the migration; behaviour, model state and
saved artifacts are unchanged. Models and PSI baselines logged by 6.x keep
loading. Old import paths are removed without compatibility aliases.

## Moved imports

| Old import | New import |
| --- | --- |
| `ml_toolkit.config.FigureConfig` | `ml_toolkit.figures.FigureConfig` |
| `ml_toolkit.plots.utils.PlotError`, `save_figure`, `numeric_vector`, `same_length`, `weights`, `feature_filename` | `ml_toolkit.figures` |
| `ml_toolkit.config.TrackingConfig` | `ml_toolkit.models.TrackingConfig` |
| `ml_toolkit.training.TrackingConfig` | `ml_toolkit.models.TrackingConfig` |
| `ml_toolkit.training.FigureConfig` | `ml_toolkit.figures.FigureConfig` |
| `ml_toolkit.training.FeatureRanker` | `ml_toolkit.feature_selection.FeatureRanker` |
| `ml_toolkit.training.Pool` | `ml_toolkit.Pool` or `ml_toolkit.data.Pool` |
| `ml_toolkit.training.contracts.DatasetSplit` | `ml_toolkit.training.DatasetSplit` |
| `ml_toolkit.training.contracts.Pool` | `ml_toolkit.Pool` or `ml_toolkit.data.Pool` |
| `ml_toolkit.training.evaluation.calculate_metrics` | `ml_toolkit.metrics.calculate_metrics` |
| `ml_toolkit.metrics._evaluation.validate_prediction` | `ml_toolkit.data.validate_prediction` |
| `ml_toolkit.data_analysis.validate_split` | `ml_toolkit.data.validate_split_labels` |
| `ml_toolkit.evaluation._inputs.combine_evaluation_pools` | `ml_toolkit.data.concat_pools` |
| `ml_toolkit._rows.row_indexer` | `ml_toolkit.data.row_indexer` |
| `ml_toolkit._model_input.categorical_frame` | `ml_toolkit.data.categorical_frame` |
| `ml_toolkit.tracking.state` | `ml_toolkit.models.state` |
| `ml_toolkit.tracking.model_artifacts` | `ml_toolkit.models.artifacts` |
| `ml_toolkit.tracking.mlflow.log_polars_table` | `ml_toolkit.tracking.log_polars_table` |
| `ml_toolkit.tracking.psi.load_psi_baseline` | `ml_toolkit.tracking.load_psi_baseline` |
| `ml_toolkit.tracking.runs.run`, `log_json` | `ml_toolkit.tracking.nested_run`, `log_json` |
| `ml_toolkit.tracking.runs.log_params`, `log_metrics`, `log_artifact` | removed; call `mlflow` directly |
| `ml_toolkit.tracking.model_artifacts.merge_artifacts` | `ml_toolkit.models.merge_artifacts` |
| `ml_toolkit.training.contracts.TrainingStage` | `ml_toolkit.training.results.TrainingStage` |
| `ml_toolkit.tracking.entrypoints.*` | removed; see below |

## Packaging entrypoints

MLflow models-from-code entrypoints are now generated when a model is logged,
so `ml_toolkit.tracking.entrypoints` no longer exists. Artifacts written by
earlier versions contain their own entrypoint file and load unchanged.

`log_model_from_code` (in `ml_toolkit.models.artifacts`) now takes `model_class`
(and optionally `model_module`, default `ml_toolkit.models`) instead of
`source`, the path of an entrypoint file. Only toolkit adapters exported from
that module can be logged this way; other classes raise `ValueError`. Custom
adapters package themselves, as in `examples/custom_model.py`.

## Internal checks

`tests/test_architecture.py` enforces the layer order between modules.

## Reports (phase 3)

`ml_toolkit.evaluation`, `ml_toolkit.model_evaluation`, `ml_toolkit.data_analysis`
and the report-related figure modules of `ml_toolkit.plots` are merged into
`ml_toolkit.reports`. Each report module holds its computation, result types and
figure functions. Importing a report class from its package keeps working:
`from ml_toolkit.reports import PrecisionRecall`.

| Old import | New import |
| --- | --- |
| `ml_toolkit.evaluation.*` (report classes, `PredictionEvaluator`, `ResidualResult`) | `ml_toolkit.reports` |
| `ml_toolkit.evaluation.comparison_pair`, `validate_inputs` | `ml_toolkit.reports.comparison_pair`, `validate_inputs` |
| `ml_toolkit.evaluation._metrics.validate_report_metrics` | `ml_toolkit.reports.validate_report_metrics` |
| `ml_toolkit.evaluation.NumericalFeatureDistributions` (prediction-based adapter) | removed; use `ml_toolkit.reports.NumericalFeatureDistributions` with `data_analyzers` |
| `ml_toolkit.model_evaluation.*` | `ml_toolkit.reports` |
| `ml_toolkit.data_analysis.*`, `data_comparison_pair` | `ml_toolkit.reports` |
| `ml_toolkit._grouping` (`row_groups`, `RowGroup`, `validate_grouping`) | `ml_toolkit.reports.grouped` |
| `ml_toolkit.plots.classification` | `ml_toolkit.reports.classification` |
| `ml_toolkit.plots.ranking` | `ml_toolkit.reports.ranking` |
| `ml_toolkit.plots.pu` | `ml_toolkit.reports.pu` |
| `ml_toolkit.plots.regression` | `ml_toolkit.reports.regression` |
| `ml_toolkit.plots.residuals` | `ml_toolkit.reports.residuals` |
| `ml_toolkit.plots.metric_trend` | `ml_toolkit.reports.metric_trend` |
| `ml_toolkit.plots.permutation_importance` | `ml_toolkit.reports.permutation` |
| `ml_toolkit.plots.robustness` | `ml_toolkit.reports.robustness` |
| `ml_toolkit.plots.model_evaluation.shrink_curve` | `ml_toolkit.reports.shrink.shrink_curve` |
| `ml_toolkit.plots.monitoring.time_feature_distribution`, `validate_distribution` | `ml_toolkit.reports.distributions` |
| `ml_toolkit.plots.explainability` | `ml_toolkit.explainability` |
| `ml_toolkit.config.ShapConfig`, `ml_toolkit.training.ShapConfig` | `ml_toolkit.explainability.ShapConfig` |

`ml_toolkit.plots` keeps only `psi` and `feature_selection` until their own
phases. Computation that used to sit inside figure functions is available
without matplotlib: `classification.precision_recall_summary`,
`classification.calibration_bins`, `regression.level_observed_fitted`,
`regression.time_summary` and `distributions.distribution_summary`.
`metrics.classification.average_precision` is the single average-precision
implementation; the precision-recall figure reuses it.
