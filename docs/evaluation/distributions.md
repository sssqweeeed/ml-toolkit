# Numerical distributions moved to Data analysis

See [Numerical distributions](../data-analysis/distributions.md) for the guide,
plots and API reference. New code imports `NumericalFeatureDistributions` from
`ml_toolkit.data_analysis` and passes it to `train_model(data_analyzers=...)`.
The previous `ml_toolkit.evaluation` import and `prediction_evaluators` usage
remain supported for existing callers.
