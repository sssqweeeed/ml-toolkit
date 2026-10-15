# Explainability and tracking

## Explainability

Import helpers from `ml_toolkit.explainability`. Install the `shap` extra for
diagnostic SHAP explanations; consult each function's requirements.

::: ml_toolkit.explainability
    options:
      members: [shap_contributions, calculate_shap_values, top_shap_features]

## MLflow tables

Configure the MLflow tracking URI and experiment in your application. Model
adapters expose their own logging methods in the [model reference](models.md).

::: ml_toolkit.tracking.log_polars_table
