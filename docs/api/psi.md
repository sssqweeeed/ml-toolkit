# PSI and feature levels

Import PSI analysis from `ml_toolkit.psi`. Fit reference levels on training data
and reuse them when comparing later samples.

::: ml_toolkit.psi

## Shared feature levels

Import these lower-level types from `ml_toolkit.feature_levels`.
`FeatureLevelSpec` is also exported from `ml_toolkit.config` and documented in the
[configuration reference](configuration.md).

::: ml_toolkit.feature_levels
    options:
      members:
        - LevelTransformer
        - LevelConfig
        - FeatureLevelConfig
        - FeatureType
        - FeatureScalar
        - NULL_LEVEL
        - ERROR_LEVEL
        - UNSEEN_LEVEL
