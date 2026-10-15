# ml-toolkit

Reusable tools for model training, chronological validation, feature selection,
Optuna tuning, metrics, diagnostic plots, and MLflow logging.

Prepare your data as a Polars-backed `Pool`, choose a model adapter and metric,
then run the same training workflow across tasks.

## Start here

- [Getting started](getting-started.md): install the package and train a model
  on a complete, reproducible example.
- [Configuration by task](configuration.md): expanded recipes for classification,
  regression, frequency, severity, and positive-unlabeled learning.
- [Training API](api/training.md): validation, feature selection, tuning, and
  result objects.
- [Model API](api/models.md): CatBoost adapters, inflation adjustment, and the
  optional LightGBM integration.

## A chronological training workflow

1. Reserve an optional future test tail.
2. For each validation split, select features, tune parameters, and fit the
   winning settings. Selection and tuning are optional.
3. Fit the final model on every non-test row using the last split's settings.
4. Evaluate the test tail after fitting and log results to MLflow.

The caller prepares features and targets, orders observations by ascending time,
and configures the MLflow tracking URI and experiment.

## More tools

- [Metrics](api/metrics.md): classification, frequency, severity, PU, and ranking.
- [PSI and feature levels](api/psi.md): reference-fitted bins and drift analysis.
- [Plots](api/plots.md): observed/fitted, ranking, calibration, and monitoring.
- [Inflation](api/inflation.md): constant-rate and monthly-index strategies.
- [Explainability and tracking](api/tracking.md): SHAP helpers and table logging.

This site follows the `main` branch. Source code and additional usage notes are
available in the [GitHub repository](https://github.com/sssqweeeed/ml-toolkit).
