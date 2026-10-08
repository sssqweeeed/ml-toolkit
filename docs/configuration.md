# Configuration by task

Configurations are immutable, keyword-only dataclasses. Public contracts are
checked statically with strict Pyright; each configuration checks its own value
ranges without reflection or runtime type discovery. Use `dataclasses.replace`
for changes; data, model factories, metrics, rankers and Optuna spaces are explicit
function arguments. The examples below expand every constructor field.

```python
from ml_toolkit.config import (
    HoldoutConfig,
    TimeSeriesCVConfig,
    RFEFeatureSelectionConfig,
    TuningConfig,
    TrainingConfig,
    TrackingConfig,
    PSIOptions,
    FigureConfig,
    NumericalFeatureDistributionConfig,
    ShapConfig,
    CommonPlotsConfig,
    ClassificationPlotsConfig,
    RegressionPlotsConfig,
    PUPlotsConfig,
    ClassificationDiagnosticsConfig,
    RegressionDiagnosticsConfig,
    PUDiagnosticsConfig,
    PostFitEvaluationConfig,
    ReportingConfig,
)
```

- `validation` is `HoldoutConfig` or `TimeSeriesCVConfig`. Both execute the same
  complete train/validation operation, including optional feature selection.
- Integers specify row counts; floats specify fractions strictly between 0 and 1,
  rounded up. Fractions are relative to the data at that split: original rows for
  test and retained non-test rows for validation. RFE uses the external train/val.
- `test_size=None` disables test evaluation. Otherwise the last rows are reserved
  before any training or optimization, and evaluated only after final fitting.
- Rows must be ordered by ascending `time`; there is no shuffle. `gap` and
  `max_train_size` are row counts. Explicit diagnostic indices use original
  one-based split numbers before `skip_splits`.
- Every split performs RFE once on its explicit train/val with base parameters,
  then fixes its features for its own Optuna study and post-tuning fit. Final
  fitting uses the last split's features, parameters, iterations and threshold.
- Each split has an independent Optuna study scoring only its own val; trial and
  time budgets apply per study. One fit with winning parameters follows each
  search. Mean validation metrics are reports, not optimization objectives.
- `reporting.validation_splits` selects built-in validation diagnostics only.
  Test uses the same diagnostic functions when enabled. Trials have no built-in
  diagnostics; custom trial evaluation can be enabled through `post_fit.stages`.
- Custom evaluation defaults to validation and final stages, with train/val/test
  enabled where available. It needs aligned `object_id` and does not contribute
  to optimization or validation averages. Images follow `write_artifacts`.
- Model packaging defaults to final fit only. Registry names apply only to that
  final model. Metrics and configuration artifacts are always logged.
- Regression ranking plots default off for signed targets. Frequency and severity
  recipes enable gains/lift and require nonnegative targets with positive totals.
- `None` disables optional stages. `features=None` selects all eligible features;
  `features=()` selects none. SHAP defaults off; enable it with `ShapConfig` and
  install the `shap` extra for diagnostic explanations.

## Binary classification

```python
classification_config = TrainingConfig(
    validation=TimeSeriesCVConfig(
        n_splits=5,
        val_size=None,
        gap=0,
        max_train_size=None,
        skip_splits=0,
    ),
    test_size=0.2,
    feature_selection=RFEFeatureSelectionConfig(
        min_features=10,
        metric='log_loss',
        direction='minimize',
        max_fits=6,
        score_tolerance=0.0,
        protected_features=(),
    ),
    tuning=TuningConfig(
        metric='log_loss',
        direction='minimize',
        n_trials=50,
        timeout_seconds=None,
        seed=42,
    ),
    reporting=ReportingConfig(
        diagnostics=ClassificationDiagnosticsConfig(
            psi=PSIOptions(
                n_bins=10,
                epsilon=1e-06,
                features=None,
            ),
            plots=ClassificationPlotsConfig(
                common=CommonPlotsConfig(
                    figure=FigureConfig(
                        figsize=None,
                        dpi=150,
                        format='png',
                    ),
                    psi_scores=True,
                    psi_distribution_features=None,
                    numerical_distributions=NumericalFeatureDistributionConfig(
                        features=None,
                        quantile_bands=((0.05, 0.95), (0.25, 0.75)),
                        show_mean=True,
                        show_median=True,
                    ),
                    shap=None,
                ),
                precision_recall=True,
                f1_threshold=True,
                calibration=True,
                gains=True,
                lift=True,
                monthly_observed_fitted=True,
                max_curve_points=1000,
                calibration_bins=10,
                histogram_bins=30,
            ),
        ),
        validation_splits='last',
        post_fit=PostFitEvaluationConfig(
            stages=('validation', 'final'),
            splits=('train', 'val', 'test'),
            write_artifacts=True,
        ),
    ),
    tracking=TrackingConfig(
        log_trial_models=False,
        log_validation_models=False,
        log_final_model=True,
        input_example_rows=5,
        model_name='model',
        registered_model_name=None,
        conda_env=None,
    ),
)
```

## Regression

```python
regression_config = TrainingConfig(
    validation=TimeSeriesCVConfig(
        n_splits=5,
        val_size=None,
        gap=0,
        max_train_size=None,
        skip_splits=0,
    ),
    test_size=0.2,
    feature_selection=RFEFeatureSelectionConfig(
        min_features=10,
        metric='mae',
        direction='minimize',
        max_fits=6,
        score_tolerance=0.0,
        protected_features=(),
    ),
    tuning=TuningConfig(
        metric='mae',
        direction='minimize',
        n_trials=50,
        timeout_seconds=None,
        seed=42,
    ),
    reporting=ReportingConfig(
        diagnostics=RegressionDiagnosticsConfig(
            task='regression',
            psi=PSIOptions(
                n_bins=10,
                epsilon=1e-06,
                features=None,
            ),
            plots=RegressionPlotsConfig(
                common=CommonPlotsConfig(
                    figure=FigureConfig(
                        figsize=None,
                        dpi=150,
                        format='png',
                    ),
                    psi_scores=True,
                    psi_distribution_features=None,
                    numerical_distributions=NumericalFeatureDistributionConfig(
                        features=None,
                        quantile_bands=((0.05, 0.95), (0.25, 0.75)),
                        show_mean=True,
                        show_median=True,
                    ),
                    shap=None,
                ),
                gains=False,
                lift=False,
                monthly_observed_fitted=True,
            ),
        ),
        validation_splits='last',
        post_fit=PostFitEvaluationConfig(
            stages=('validation', 'final'),
            splits=('train', 'val', 'test'),
            write_artifacts=True,
        ),
    ),
    tracking=TrackingConfig(
        log_trial_models=False,
        log_validation_models=False,
        log_final_model=True,
        input_example_rows=5,
        model_name='model',
        registered_model_name=None,
        conda_env=None,
    ),
)
```

## Frequency

```python
frequency_config = TrainingConfig(
    validation=TimeSeriesCVConfig(
        n_splits=5,
        val_size=None,
        gap=0,
        max_train_size=None,
        skip_splits=0,
    ),
    test_size=0.2,
    feature_selection=RFEFeatureSelectionConfig(
        min_features=10,
        metric='mean_poisson_deviance',
        direction='minimize',
        max_fits=6,
        score_tolerance=0.0,
        protected_features=(),
    ),
    tuning=TuningConfig(
        metric='mean_poisson_deviance',
        direction='minimize',
        n_trials=50,
        timeout_seconds=None,
        seed=42,
    ),
    reporting=ReportingConfig(
        diagnostics=RegressionDiagnosticsConfig(
            task='frequency',
            psi=PSIOptions(
                n_bins=10,
                epsilon=1e-06,
                features=None,
            ),
            plots=RegressionPlotsConfig(
                common=CommonPlotsConfig(
                    figure=FigureConfig(
                        figsize=None,
                        dpi=150,
                        format='png',
                    ),
                    psi_scores=True,
                    psi_distribution_features=None,
                    numerical_distributions=NumericalFeatureDistributionConfig(
                        features=None,
                        quantile_bands=((0.05, 0.95), (0.25, 0.75)),
                        show_mean=True,
                        show_median=True,
                    ),
                    shap=None,
                ),
                gains=True,
                lift=True,
                monthly_observed_fitted=True,
            ),
        ),
        validation_splits='last',
        post_fit=PostFitEvaluationConfig(
            stages=('validation', 'final'),
            splits=('train', 'val', 'test'),
            write_artifacts=True,
        ),
    ),
    tracking=TrackingConfig(
        log_trial_models=False,
        log_validation_models=False,
        log_final_model=True,
        input_example_rows=5,
        model_name='model',
        registered_model_name=None,
        conda_env=None,
    ),
)
```

## Severity

```python
severity_config = TrainingConfig(
    validation=TimeSeriesCVConfig(
        n_splits=5,
        val_size=None,
        gap=0,
        max_train_size=None,
        skip_splits=0,
    ),
    test_size=0.2,
    feature_selection=RFEFeatureSelectionConfig(
        min_features=10,
        metric='mean_tweedie_deviance',
        direction='minimize',
        max_fits=6,
        score_tolerance=0.0,
        protected_features=(),
    ),
    tuning=TuningConfig(
        metric='mean_tweedie_deviance',
        direction='minimize',
        n_trials=50,
        timeout_seconds=None,
        seed=42,
    ),
    reporting=ReportingConfig(
        diagnostics=RegressionDiagnosticsConfig(
            task='severity',
            psi=PSIOptions(
                n_bins=10,
                epsilon=1e-06,
                features=None,
            ),
            plots=RegressionPlotsConfig(
                common=CommonPlotsConfig(
                    figure=FigureConfig(
                        figsize=None,
                        dpi=150,
                        format='png',
                    ),
                    psi_scores=True,
                    psi_distribution_features=None,
                    numerical_distributions=NumericalFeatureDistributionConfig(
                        features=None,
                        quantile_bands=((0.05, 0.95), (0.25, 0.75)),
                        show_mean=True,
                        show_median=True,
                    ),
                    shap=None,
                ),
                gains=True,
                lift=True,
                monthly_observed_fitted=True,
            ),
        ),
        validation_splits='last',
        post_fit=PostFitEvaluationConfig(
            stages=('validation', 'final'),
            splits=('train', 'val', 'test'),
            write_artifacts=True,
        ),
    ),
    tracking=TrackingConfig(
        log_trial_models=False,
        log_validation_models=False,
        log_final_model=True,
        input_example_rows=5,
        model_name='model',
        registered_model_name=None,
        conda_env=None,
    ),
)
```

## Positive–unlabeled classification

```python
pu_classification_config = TrainingConfig(
    validation=TimeSeriesCVConfig(
        n_splits=5,
        val_size=None,
        gap=0,
        max_train_size=None,
        skip_splits=0,
    ),
    test_size=0.2,
    feature_selection=RFEFeatureSelectionConfig(
        min_features=10,
        metric='pu_risk',
        direction='minimize',
        max_fits=6,
        score_tolerance=0.0,
        protected_features=(),
    ),
    tuning=TuningConfig(
        metric='pu_risk',
        direction='minimize',
        n_trials=50,
        timeout_seconds=None,
        seed=42,
    ),
    reporting=ReportingConfig(
        diagnostics=PUDiagnosticsConfig(
            psi=PSIOptions(
                n_bins=10,
                epsilon=1e-06,
                features=None,
            ),
            plots=PUPlotsConfig(
                common=CommonPlotsConfig(
                    figure=FigureConfig(
                        figsize=None,
                        dpi=150,
                        format='png',
                    ),
                    psi_scores=True,
                    psi_distribution_features=None,
                    numerical_distributions=NumericalFeatureDistributionConfig(
                        features=None,
                        quantile_bands=((0.05, 0.95), (0.25, 0.75)),
                        show_mean=True,
                        show_median=True,
                    ),
                    shap=None,
                ),
                score_distribution=True,
                positive_capture=True,
                histogram_bins=30,
            ),
        ),
        validation_splits='last',
        post_fit=PostFitEvaluationConfig(
            stages=('validation', 'final'),
            splits=('train', 'val', 'test'),
            write_artifacts=True,
        ),
    ),
    tracking=TrackingConfig(
        log_trial_models=False,
        log_validation_models=False,
        log_final_model=True,
        input_example_rows=5,
        model_name='model',
        registered_model_name=None,
        conda_env=None,
    ),
)
```

## Switching validation strategies

```python
from dataclasses import replace
holdout_config = replace(regression_config, validation=HoldoutConfig(val_size=0.2))
no_test_config = replace(holdout_config, test_size=None)
```

The training and Optuna calls remain identical. The final model fits every retained
training row, including former validation and gap rows. `result.validation` holds
split metadata and average val metrics; `result.test_metrics` is `None` when test
is disabled. `result.params` and `result.features` come from the last split's
post-tuning fit, rather than a median or a feature union.
