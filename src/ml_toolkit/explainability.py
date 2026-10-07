"""Prepare SHAP explanations from supported native model backends."""

import logging

import numpy as np
import polars as pl

from .config import ShapConfig

logger = logging.getLogger(__name__)


def _shap_contributions(model, data, cat_features=()):
    import catboost

    if isinstance(model, catboost.CatBoost):
        names = model.feature_names_
        if not names:
            raise ValueError("Fit the model before calculating SHAP")
        categorical = [names[index] for index in model.get_cat_feature_indices()]
        features = (
            data.select(names)
            .with_columns(
                pl.col(name).cast(pl.String).fill_null("unknown")
                for name in categorical
            )
            .to_pandas()
        )
        contributions = np.asarray(
            model.get_feature_importance(
                catboost.Pool(features, cat_features=model.get_cat_feature_indices()),
                type="ShapValues",
            )
        )
    else:
        try:
            import lightgbm
        except (ImportError, OSError) as error:
            raise ImportError(
                "LightGBM SHAP requires a working LightGBM installation"
            ) from error
        if not isinstance(model, lightgbm.Booster):
            raise TypeError(f"Unsupported model for SHAP: {type(model).__name__}")
        names = model.feature_name()
        features = data.select(names).to_pandas()
        for name in features.columns:
            if name in cat_features or features[name].dtype.name in (
                "object", "string", "category"
            ):
                features[name] = features[name].astype("category")
        contributions = np.asarray(model.predict(features, pred_contrib=True))
    if contributions.shape != (len(data), len(names) + 1):
        raise ValueError("Only single-output SHAP contributions are supported")
    return names, features, contributions


def calculate_shap_values(
    model, data: pl.DataFrame, *, config: ShapConfig | None = None
):
    """Calculate a single-output SHAP explanation from a native tree model.

    Parameters
    ----------
    model : catboost.CatBoost or lightgbm.Booster
        Fitted native model, rather than a toolkit adapter or project Pool.
    data : polars.DataFrame
        Nonempty feature frame containing the native model feature names.
    config : ShapConfig or None, default None
        Maximum sampled rows and sampling seed. None uses ShapConfig defaults.
        Display settings are consumed by plotting, not this calculation.

    Returns
    -------
    shap.Explanation
        Sampled row contributions, row base values, feature data, and model
        feature names. Contributions are on the native raw-output scale.

    Raises
    ------
    ImportError
        The shap extra or a working LightGBM installation is unavailable.
    TypeError
        The native model backend is unsupported.
    ValueError
        Data is empty, the model lacks fitted feature
        names, or contributions are not single-output.
    """
    try:
        import shap
    except ImportError as error:
        raise ImportError(
            "Install ml-toolkit[shap] to calculate SHAP values"
        ) from error
    if not len(data):
        raise ValueError("Cannot calculate SHAP values for an empty dataset")
    config = ShapConfig() if config is None else config
    sampled = data.sample(min(config.max_samples, len(data)), seed=config.seed)
    logger.debug(
        "Calculating SHAP for %s: %s sampled rows", type(model).__name__, len(sampled)
    )
    names, features, contributions = _shap_contributions(model, sampled)
    logger.debug("Calculated SHAP: %s rows, %s features", len(sampled), len(names))
    return shap.Explanation(
        values=contributions[:, :-1],
        base_values=contributions[:, -1],
        data=features.to_numpy(),
        feature_names=list(names),
    )


def top_shap_features(values, top_n: int) -> list[str]:
    """Rank feature names by descending mean absolute SHAP contribution.

    Parameters
    ----------
    values : shap.Explanation
        Single-output explanation with one feature name per contribution
        column.
    top_n : int
        Positive maximum number of feature names to return.

    Returns
    -------
    list[str]
        Up to top_n feature names, with ties preserving original column order.

    Raises
    ------
    ValueError
        top_n is nonpositive or feature names are missing or misaligned.
    """
    if top_n < 1:
        raise ValueError("top_n must be positive")
    importance = np.asarray(values.abs.mean(0).values)
    if values.feature_names is None or len(values.feature_names) != len(importance):
        raise ValueError("SHAP explanation must have feature names")
    return [
        values.feature_names[index]
        for index in np.argsort(-importance, kind="stable")[:top_n]
    ]
