# Migrating to 7.0

Version 7.0 reorganizes the package so that every concept has one home. Mostly
import paths change; the few behaviour changes are listed under "Behaviour
changes" (and per phase below). Model state and saved artifacts are unchanged.
Models and PSI baselines logged by 6.x keep loading. Old import paths are removed without compatibility aliases.

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
| `ml_toolkit.tracking.psi.load_psi_baseline` | `ml_toolkit.psi.load_psi_baseline` |
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

## Models (phase 2)

Most of these are internal modules. The classes exported from
`ml_toolkit.models` and the model-state schema are unchanged, except that two
methods of the CatBoost adapters are no longer public.

| Old import | New import |
| --- | --- |
| `CatBoostRegressionModel.log_artifacts` | removed from the public API (now private); use `log` |
| `CatBoostRegressionModel.restore_state` | removed from the public API (now private); load models through MLflow |
| `ml_toolkit.models._shrink.validate_size` | `ml_toolkit.models.base.validate_size` |
| `ml_toolkit.models.losses.gamma.config`, `.gamma.cpu` | `ml_toolkit.models.losses.gamma` |
| `ml_toolkit.models.losses.gamma.cuda` | `ml_toolkit.models.losses.gamma_cuda` |
| `ml_toolkit.models.losses.pu.common`, `.pu.cpu` | `ml_toolkit.models.losses.pu` |
| `ml_toolkit.models.losses.pu.cuda` | `ml_toolkit.models.losses.pu_cuda` |
| `ml_toolkit.models.losses.cuda` | `ml_toolkit.models.losses._cuda` (private) |
| `UPUMetric`, `NNPUMetric` from `ml_toolkit.models.catboost_classification` | `ml_toolkit.models.losses.pu` |

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

`ml_toolkit.plots` is removed; every figure function lives next to its concept.
Computation that used to sit inside figure functions is available
without matplotlib: `classification.precision_recall_summary`,
`classification.calibration_bins`, `regression.level_observed_fitted`,
`regression.time_summary` and `distributions.distribution_summary`.
`metrics.classification.average_precision` is the single average-precision
implementation; the precision-recall figure reuses it.

## Feature selection (phase 5)

`ml_toolkit.feature_selection` is now a package. Its public names are
exported from `ml_toolkit.feature_selection` itself.

| Old import | New import |
| --- | --- |
| `ml_toolkit.training.importance_ranker` | `ml_toolkit.feature_selection.importance_ranker` |
| `ml_toolkit.training.ShapRanker` | `ml_toolkit.feature_selection.ShapRanker` |
| `ml_toolkit.training.rankers` | `ml_toolkit.feature_selection` |
| `ml_toolkit.plots.feature_selection.*` (`rfe_progress`, `top_k_ranking`, `forward_progress`, `forward_improvement`, `best_iteration_progress`) | `ml_toolkit.feature_selection.figures` |
| `ml_toolkit.feature_selection.SelectionSettings` (a `dict`) | `ml_toolkit.feature_selection.SelectionSettings`, a union of `RFESettings`, `TopKSettings`, `ForwardSettings` and `CustomSelectionSettings` |

Custom selectors return `CustomSelectionSettings("name", {"key": value})`
instead of `{"selector": "name", "key": value}`. `FeatureSelectionResult.settings`
defaults to `None` instead of `{}`.

## Behaviour changes

- `FeatureSelectionResult.settings` is a typed dataclass: use attribute access
  (`result.settings.k`, `result.settings.evaluator`) instead of dictionary
  access. `evaluator` is `None` when unset. `selection/history.json` is
  unchanged; `FeatureSelectionResult.to_dict()` is the only way to build the
  logged payload (`dataclasses.asdict` nests `details` and keeps
  `evaluator: null`). Settings reject negative or non-finite numeric fields,
  and `CustomSelectionSettings` rejects the built-in selector names and a
  `selector` key in `details`.
- `GroupedMetrics` now reports a metric callback that returns no scores or
  empty names as `Metric must return nonempty named scores` (previously
  `Grouped metric must return at least one named score` or `Grouped metric
  names must be nonempty strings`), the same message `MetricTrend` and
  `ShrinkCurve` use.
- `classification.calibration` reports a nonpositive `histogram_bins` as
  `histogram_bins must be positive`; a nonpositive `n_bins` is rejected by
  `classification.calibration_bins` as `n_bins must be positive` (previously
  both gave `bin counts must be positive`).

## PSI and feature levels (phase 4)

`ml_toolkit.feature_levels` is a single module and the `ml_toolkit.psi`
submodules are merged. Names exported from `ml_toolkit.psi` and
`ml_toolkit.feature_levels` keep their import paths. PSI numbers, saved
baselines and `LevelTransformer` state are unchanged.

| Old import | New import |
| --- | --- |
| `ml_toolkit.feature_levels.config.*` (`FeatureLevelConfig`, `FeatureLevels`, `NULL_LEVEL`, ..., `feature_definitions`, `validate_features`) | `ml_toolkit.feature_levels` |
| `ml_toolkit.feature_levels.levels.LevelTransformer` | `ml_toolkit.feature_levels.LevelTransformer` |
| `ml_toolkit.feature_levels._serialization.encode_scalar`, `decode_scalar`, `mapping` | `ml_toolkit.feature_levels.encode_feature_scalar`, `decode_feature_scalar`, `json_object` |
| `ml_toolkit.psi.config.PSIConfig`, `ml_toolkit.psi.results.PSIResult` | `ml_toolkit.psi.PSIConfig`, `ml_toolkit.psi.PSIResult` |
| `ml_toolkit.psi.distributions.*` | `ml_toolkit.psi.analysis` |
| `ml_toolkit.tracking.load_psi_baseline` | `ml_toolkit.psi.load_psi_baseline` |
| `ml_toolkit.plots.psi` | `ml_toolkit.psi.figures` |
| `ml_toolkit.training.psi.prepare_psi_baseline`, `log_model_with_baseline` | `ml_toolkit.training.reporting` |

## Training workflow (phase 6)

`train_model` is split into named internal steps; training results, logged
MLflow parameters, metrics and artifacts, and console output are unchanged.
Each public name now has one import path.

| Old import | New import |
| --- | --- |
| `ml_toolkit.training.TuningConfig`, `TrainingConfig`, `ReportingConfig` | `ml_toolkit.config` |
| `ml_toolkit.training.FeatureLevelConfig`, `ml_toolkit.config.FeatureLevelConfig` | `ml_toolkit.feature_levels.FeatureLevelConfig` |
| `ml_toolkit.config.TimePeriod` | `ml_toolkit.types.TimePeriod` |
| `ml_toolkit.data.validate_auxiliary_pair(left, right)` | `ml_toolkit.data.validate_compatible_pools(left, right, context=...)` |

Behaviour changes:

- One check, `validate_compatible_pools`, compares feature schemas (names,
  order and dtypes), categorical and numerical feature declarations and
  auxiliary schemas. `Split`, `concat_pools`, validation fits, feature
  selection and the `fit_on_test` final fit all use it, so the messages now read
  `<subject> must have matching features and metadata` (for example `Split
  pools`, `Reporting pools`, `Train and validation pools`, `Final-fit data`)
  or `... must have matching auxiliary feature schemas`; the feature check runs
  first, so a pair that differs in both reports the feature message. Feature selection and
  validation fits previously compared column names only; they now also reject
  differing dtypes or feature declarations between train and validation.
