# Plots

Import plotting modules from `ml_toolkit.plots`, for example
`from ml_toolkit.plots import classification`. Plot functions save figures to
the requested paths; some also return the underlying tables. See each function
for target, weight, and output-path constraints.

::: ml_toolkit.plots.classification
    options:
      members: [binary_curves, precision_recall, f1_threshold, calibration]

::: ml_toolkit.plots.regression
    options:
      members: [observed_fitted, observed_fitted_by_level, monthly_observed_fitted]

::: ml_toolkit.plots.ranking
    options:
      members: [gains_curve, lift_curve]

::: ml_toolkit.plots.pu
    options:
      members: [score_distribution, positive_capture]

::: ml_toolkit.plots.psi
    options:
      members: [feature_psi_plot, monthly_psi_level_distribution]

::: ml_toolkit.plots.monitoring
    options:
      members: [monthly_feature_distribution]

::: ml_toolkit.plots.explainability
    options:
      members: [feature_importance, shap_importance, shap_beeswarm, shap_dependence]
