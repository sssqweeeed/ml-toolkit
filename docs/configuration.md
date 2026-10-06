# Configuration by task

One fully expanded configuration per supported task. Every constructor field is
written explicitly, including nested defaults. These are examples, not global
presets: use `dataclasses.replace` to change a recipe without mutating it.

```python
from ml_toolkit.config import (
    ClassificationDiagnosticsConfig,
    ClassificationPlotsConfig, CommonPlotsConfig, CrossValidationConfig,
    FeatureSelectionConfig, FigureConfig, FinalFitConfig,
    NumericalFeatureDistributionConfig, PostFitEvaluationConfig,
    PUDiagnosticsConfig, PUPlotsConfig, PSIOptions,
    RegressionDiagnosticsConfig, RegressionPlotsConfig, ShapConfig,
    TrackingConfig, TrainingConfig, TuningConfig,
)
```

- CV sizes and `gap` are **rows**, not days; timestamps must be ascending.
- Feature selection uses only training rows of the first evaluated CV fold.
  `n_features=10` assumes at least ten input features. Every model, including
  PU, uses one fitted native model for top-k importance selection.
- `None` disables optional stages; `features=None` selects all eligible features,
  whereas `features=()` selects none. PSI definitions must be nonempty when
  explicitly supplied; use `psi=None` to disable PSI.
- `figsize=None` preserves each figure's native or adaptive size.
- SHAP is enabled in these examples to show every field; it requires `.[shap]`.
  `dependence_features=()` disables dependence plots; importance and beeswarm
  remain enabled. Use `shap=None` to disable all SHAP work.
- Ranking defaults off for ordinary regression, which can have signed targets.
  Explicit gains/lift require nonnegative validation targets and a positive
  weighted observed total. Invalid enabled plots raise instead of being skipped.
- Tuning is enabled here; supply `param_space` and a metric returning the objective
  key shown below. Model, tuning, and SHAP seeds are independent.
- Custom evaluation follows `post_fit` only when an evaluator is supplied; enabled
  calls need aligned `object_id`. Final fitting evaluates only the train split.
- Metrics and configuration artifacts are always logged. Model packaging defaults
  to final fit only. Registration applies only to that final model.

Classification and PU diagnostics have a fixed task determined by their class.
Regression diagnostics label `regression`, `frequency`, or `severity` explicitly.
All fields have NumPy-style documentation in `ml_toolkit.config` and are validated
when constructed. Passing PU diagnostics a supervised plots object is an error.

## Supervised binary classification

```python
classification_config = TrainingConfig(
    cv=CrossValidationConfig(
        n_splits=5,
        skip_splits=0,
        gap=0,
        test_size=None,
        max_train_size=None,
    ),
    feature_selection=FeatureSelectionConfig(
        n_features=10,
        validation_fraction=0.25,
    ),
    tuning=TuningConfig(
        metric='log_loss',
        direction='minimize',
        n_trials=50,
        timeout_seconds=None,
        seed=42,
    ),
    final_fit=FinalFitConfig(
        iteration_policy='cv_median',
    ),
    diagnostics=ClassificationDiagnosticsConfig(
        folds='last',
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
                shap=ShapConfig(
                    max_samples=10000,
                    seed=42,
                    max_display=10,
                    dependence_features=(),
                ),
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
    tracking=TrackingConfig(
        log_trial_models=False,
        log_fold_models=False,
        log_final_model=True,
        input_example_rows=5,
        model_name='model',
        registered_model_name=None,
        conda_env=None,
    ),
    post_fit=PostFitEvaluationConfig(
        stages=('cv', 'final'),
        splits=('train', 'val'),
        write_artifacts=True,
    ),
)
```

## Positive-unlabeled classification

```python
pu_classification_config = TrainingConfig(
    cv=CrossValidationConfig(
        n_splits=5,
        skip_splits=0,
        gap=0,
        test_size=None,
        max_train_size=None,
    ),
    feature_selection=FeatureSelectionConfig(
        n_features=10,
        validation_fraction=0.25,
    ),
    tuning=TuningConfig(
        metric='pu_risk',
        direction='minimize',
        n_trials=50,
        timeout_seconds=None,
        seed=42,
    ),
    final_fit=FinalFitConfig(
        iteration_policy='cv_median',
    ),
    diagnostics=PUDiagnosticsConfig(
        folds='last',
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
                shap=ShapConfig(
                    max_samples=10000,
                    seed=42,
                    max_display=10,
                    dependence_features=(),
                ),
            ),
            score_distribution=True,
            positive_capture=True,
            histogram_bins=30,
        ),
    ),
    tracking=TrackingConfig(
        log_trial_models=False,
        log_fold_models=False,
        log_final_model=True,
        input_example_rows=5,
        model_name='model',
        registered_model_name=None,
        conda_env=None,
    ),
    post_fit=PostFitEvaluationConfig(
        stages=('cv', 'final'),
        splits=('train', 'val'),
        write_artifacts=True,
    ),
)
```

## Frequency

```python
frequency_config = TrainingConfig(
    cv=CrossValidationConfig(
        n_splits=5,
        skip_splits=0,
        gap=0,
        test_size=None,
        max_train_size=None,
    ),
    feature_selection=FeatureSelectionConfig(
        n_features=10,
        validation_fraction=0.25,
    ),
    tuning=TuningConfig(
        metric='mean_poisson_deviance',
        direction='minimize',
        n_trials=50,
        timeout_seconds=None,
        seed=42,
    ),
    final_fit=FinalFitConfig(
        iteration_policy='cv_median',
    ),
    diagnostics=RegressionDiagnosticsConfig(
        task='frequency',
        folds='last',
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
                shap=ShapConfig(
                    max_samples=10000,
                    seed=42,
                    max_display=10,
                    dependence_features=(),
                ),
            ),
            gains=True,
            lift=True,
            monthly_observed_fitted=True,
        ),
    ),
    tracking=TrackingConfig(
        log_trial_models=False,
        log_fold_models=False,
        log_final_model=True,
        input_example_rows=5,
        model_name='model',
        registered_model_name=None,
        conda_env=None,
    ),
    post_fit=PostFitEvaluationConfig(
        stages=('cv', 'final'),
        splits=('train', 'val'),
        write_artifacts=True,
    ),
)
```

## Severity

```python
severity_config = TrainingConfig(
    cv=CrossValidationConfig(
        n_splits=5,
        skip_splits=0,
        gap=0,
        test_size=None,
        max_train_size=None,
    ),
    feature_selection=FeatureSelectionConfig(
        n_features=10,
        validation_fraction=0.25,
    ),
    tuning=TuningConfig(
        metric='mean_tweedie_deviance',
        direction='minimize',
        n_trials=50,
        timeout_seconds=None,
        seed=42,
    ),
    final_fit=FinalFitConfig(
        iteration_policy='cv_median',
    ),
    diagnostics=RegressionDiagnosticsConfig(
        task='severity',
        folds='last',
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
                shap=ShapConfig(
                    max_samples=10000,
                    seed=42,
                    max_display=10,
                    dependence_features=(),
                ),
            ),
            gains=True,
            lift=True,
            monthly_observed_fitted=True,
        ),
    ),
    tracking=TrackingConfig(
        log_trial_models=False,
        log_fold_models=False,
        log_final_model=True,
        input_example_rows=5,
        model_name='model',
        registered_model_name=None,
        conda_env=None,
    ),
    post_fit=PostFitEvaluationConfig(
        stages=('cv', 'final'),
        splits=('train', 'val'),
        write_artifacts=True,
    ),
)
```

## Ordinary regression

```python
regression_config = TrainingConfig(
    cv=CrossValidationConfig(
        n_splits=5,
        skip_splits=0,
        gap=0,
        test_size=None,
        max_train_size=None,
    ),
    feature_selection=FeatureSelectionConfig(
        n_features=10,
        validation_fraction=0.25,
    ),
    tuning=TuningConfig(
        metric='mae',
        direction='minimize',
        n_trials=50,
        timeout_seconds=None,
        seed=42,
    ),
    final_fit=FinalFitConfig(
        iteration_policy='cv_median',
    ),
    diagnostics=RegressionDiagnosticsConfig(
        task='regression',
        folds='last',
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
                shap=ShapConfig(
                    max_samples=10000,
                    seed=42,
                    max_display=10,
                    dependence_features=(),
                ),
            ),
            gains=False,
            lift=False,
            monthly_observed_fitted=True,
        ),
    ),
    tracking=TrackingConfig(
        log_trial_models=False,
        log_fold_models=False,
        log_final_model=True,
        input_example_rows=5,
        model_name='model',
        registered_model_name=None,
        conda_env=None,
    ),
    post_fit=PostFitEvaluationConfig(
        stages=('cv', 'final'),
        splits=('train', 'val'),
        write_artifacts=True,
    ),
)
```

## Use a configuration

Data, native parameters, model factories, metric callables, parameter search
functions, and evaluators are separate dependencies:

```python
from ml_toolkit.training import train_model

result = train_model(
    data=data,
    params=model_params,
    model_factory=model_factory,
    metric=metric,
    param_space=param_space,
    post_fit_evaluator=evaluator,  # Optional; omit when unused.
    run_name="severity",
    config=severity_config,
)
```

For PU, use matching model `loss`/`pu_prior` and `PUMetric(prior, loss=...)`.
For severity, bind the Tweedie `power` in the metric, for example with
`functools.partial(metrics_severity, power=2)` for positive Gamma targets.
For ordinary regression, the supplied metric must return `mae`. Choose an adapter
suitable for your target: `CatBoostRegressionModel` keeps native initialization
and prediction semantics and accepts signed targets with losses such as RMSE.
For Poisson/Tweedie with a manual log-mean baseline use
`CatBoostLogLinkRegressionModel`. Binary classification uses
`CatBoostClassificationModel`; PU uses `CatBoostPUClassificationModel`.
Both regressors can be composed inside `InflationAdjustedModel`.

Standalone `cross_validate` accepts `config=CrossValidationConfig(...)`,
`tune_optuna` accepts `config=TuningConfig(...)` plus `cv=CrossValidationConfig(...)`,
and feature selection is configured only through
`TrainingConfig(feature_selection=FeatureSelectionConfig(...))`.
Plotting functions accept the same family config and optional `FigureConfig`;
`calculate_shap_values` accepts `config=ShapConfig(...)`.
