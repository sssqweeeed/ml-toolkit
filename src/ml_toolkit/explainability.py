"""Prepare SHAP explanations from supported native model backends."""

import logging

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


def calculate_shap_values(model, data: pl.DataFrame, *, n_samples: int = 100_000):
    """Return a SHAP Explanation; no model-specific Pool is accepted as input."""
    try:
        import shap
    except ImportError as error:
        raise ImportError("Install ml-toolkit[shap] to calculate SHAP values") from error
    if not len(data):
        raise ValueError("Cannot calculate SHAP values for an empty dataset")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    sampled = data.sample(min(n_samples, len(data)), seed=0)
    logger.debug("Calculating SHAP for %s: %s sampled rows", type(model).__name__, len(sampled))
    import catboost

    if isinstance(model, catboost.CatBoost):
        names = model.feature_names_
        if not names:
            raise ValueError("Fit the model before calculating SHAP")
        features = sampled.select(names).to_pandas()
        contributions = np.asarray(model.get_feature_importance(
            catboost.Pool(features, cat_features=model.get_cat_feature_indices()),
            type="ShapValues",
        ))
    else:
        try:
            import lightgbm
        except (ImportError, OSError) as error:
            raise ImportError("LightGBM SHAP requires a working LightGBM installation") from error
        if not isinstance(model, lightgbm.Booster):
            raise TypeError(f"Unsupported model for SHAP: {type(model).__name__}")
        names = model.feature_name()
        features = sampled.select(names).to_pandas()
        contributions = np.asarray(model.predict(features, pred_contrib=True))
    if contributions.ndim != 2 or contributions.shape[1] != len(names) + 1:
        raise ValueError("Only single-output SHAP contributions are supported")
    logger.debug("Calculated SHAP: %s rows, %s features", len(sampled), len(names))
    return shap.Explanation(
        values=contributions[:, :-1], base_values=contributions[:, -1],
        data=features.to_numpy(), feature_names=list(names),
    )


def top_shap_features(values, top_n: int) -> list[str]:
    if top_n < 1:
        raise ValueError("top_n must be positive")
    importance = np.asarray(values.abs.mean(0).values)
    if values.feature_names is None or len(values.feature_names) != len(importance):
        raise ValueError("SHAP explanation must have feature names")
    return [values.feature_names[index] for index in np.argsort(-importance, kind="stable")[:top_n]]
