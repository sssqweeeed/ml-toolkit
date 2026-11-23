"""Feature importance and SHAP explanations with their figures."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import catboost
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from numpy.typing import ArrayLike

from ._console import Logger
from .figures import (
    FigureConfig,
    numeric_vector,
    same_length,
    save_figure,
    saved_figure,
)
from .types import FloatArray

__all__ = [
    "ShapConfig",
    "calculate_shap_values",
    "feature_importance",
    "shap_beeswarm",
    "shap_contributions",
    "shap_dependence",
    "shap_importance",
    "top_shap_features",
]

if TYPE_CHECKING:
    import lightgbm
    import shap


logger = Logger(__name__)


@dataclass(frozen=True, kw_only=True, slots=True)
class ShapConfig:
    """Configure SHAP sampling and displayed feature contributions.

    Parameters
    ----------
    max_samples : int, default 10000
        Positive maximum row count sampled without replacement from validation or test.
    seed : int, default 0
        Nonnegative sampling seed, independent of model and tuning seeds.
    max_display : int, default 10
        Positive feature limit for importance and beeswarm plots.
    dependence_features : tuple[str, ...], default ()
        Unique feature names for dependence plots. Empty disables them.

    Notes
    -----
    Requires the shap extra and a supported single-output native tree model.
    Explanations use the native raw-output scale.
    """

    max_samples: int = 10_000
    seed: int = 0
    max_display: int = 10
    dependence_features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.max_samples < 1:
            raise ValueError("max_samples must be positive")
        if self.max_display < 1:
            raise ValueError("max_display must be positive")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        names = self.dependence_features
        if any(not name.strip() for name in names) or len(set(names)) != len(names):
            raise ValueError("dependence_features must contain unique nonempty names")


def shap_contributions(
    model: catboost.CatBoost | lightgbm.Booster,
    data: pl.DataFrame,
    cat_features: Sequence[str] = (),
) -> tuple[list[str], pd.DataFrame, FloatArray]:
    """Calculate native SHAP contributions without importing the shap package.

    Parameters
    ----------
    model : catboost.CatBoost or lightgbm.Booster
        Fitted single-output native model. Pass an adapter's native_model.
    data : polars.DataFrame
        Features containing the fitted model's columns. Extra columns are
        ignored and fitted feature order is restored.
    cat_features : sequence of str, default ()
        Additional columns to cast to pandas categorical dtype for LightGBM.
        String, object, and categorical columns are also converted. CatBoost
        uses its fitted categorical indices and replaces null categories with
        "unknown".

    Returns
    -------
    names : list[str]
        Feature names in fitted model order.
    features : pandas.DataFrame
        Backend-ready feature values in the original row order.
    contributions : FloatArray of shape (n_rows, n_features + 1)
        Per-feature contributions followed by the base value in the last
        column. Each row sums to the native raw score, before link functions
        or probability transformations.

    Raises
    ------
    ValueError
        The CatBoost model has no fitted feature names or contributions are
        not single-output.
    """
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
        names = model.feature_name()
        features = data.select(names).to_pandas()
        for name in features.columns:
            if name in cat_features or features[name].dtype.name in (
                "object",
                "string",
                "category",
            ):
                features[name] = features[name].astype("category")
        contributions = np.asarray(model.predict(features, pred_contrib=True))
    if contributions.shape != (len(data), len(names) + 1):
        raise ValueError("Only single-output SHAP contributions are supported")
    return names, features, contributions


def calculate_shap_values(
    model: catboost.CatBoost | lightgbm.Booster,
    data: pl.DataFrame,
    *,
    config: ShapConfig | None = None,
) -> shap.Explanation:
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
        The optional shap package is unavailable.
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
    names, features, contributions = shap_contributions(model, sampled)
    logger.debug("Calculated SHAP: %s rows, %s features", len(sampled), len(names))
    return shap.Explanation(
        values=contributions[:, :-1],
        base_values=contributions[:, -1],
        data=features.to_numpy(),
        feature_names=list(names),
    )


def top_shap_features(values: shap.Explanation, top_n: int) -> list[str]:
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


def feature_importance(
    feature_names: list[str],
    importance: ArrayLike,
    save_fig: str | Path,
    *,
    max_features_display: int = 10,
    figure: FigureConfig | None = None,
) -> pd.DataFrame:
    """Save supplied feature importance values in descending order.

    Parameters
    ----------
    feature_names : list[str]
        Names aligned with importance values.
    importance : array-like
        Nonempty finite numerical importance values; no backend model is
        required.
    save_fig : str or pathlib.Path
        Output figure path; parent directories are created.
    max_features_display : int, default 10
        Positive maximum number of features shown.

    figure : FigureConfig or None, default None
        Optional export dimensions, DPI, and format. None preserves native
        dimensions and the requested path format.

    Returns
    -------
    pandas.DataFrame
        Displayed feature and importance columns, sorted descending with stable
        ties and a reset row index.

    Raises
    ------
    PlotError
        Importance values are invalid or do not match feature names.
    ValueError
        max_features_display is nonpositive.

    Notes
    -----
    Creates parent directories for save_fig and closes the figure after saving.
    """
    values = numeric_vector(importance, "importance")
    names = list(feature_names)
    same_length(feature_names=names, importance=values)
    if max_features_display < 1:
        raise ValueError("max_features_display must be positive")
    ordered = (
        pd.DataFrame({"feature": names, "importance": values})
        .sort_values("importance", ascending=False, kind="stable")
        .head(max_features_display)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    with saved_figure(fig, save_fig, config=figure):
        ax.barh(ordered.feature.iloc[::-1], ordered.importance.iloc[::-1])
        ax.set(
            xlabel="Feature importance", ylabel="Feature", title="Feature importance"
        )
        fig.tight_layout()
    return ordered.reset_index(drop=True)


def shap_importance(
    values: shap.Explanation,
    save_fig: str | Path,
    *,
    config: ShapConfig | None = None,
    figure: FigureConfig | None = None,
) -> None:
    """Save a SHAP bar plot of mean absolute feature contributions.

    Parameters
    ----------
    values : shap.Explanation
        Single-output explanation, such as calculate_shap_values returns.
    save_fig : str or pathlib.Path
        Output figure path; parent directories are created.
    config : ShapConfig or None, default None
        Maximum displayed feature count. None uses ShapConfig defaults.
        Sampling settings apply during explanation calculation.

    figure : FigureConfig or None, default None
        Optional export dimensions, DPI, and format. None preserves native
        dimensions and the requested path format.

    Returns
    -------
    None
        The figure is saved to disk; no table is returned.

    Raises
    ------
    ImportError
        The optional shap extra is unavailable.

    Notes
    -----
    Creates parent directories for save_fig and closes the figure after saving.
    Requires ml-toolkit[shap].
    """
    config = ShapConfig() if config is None else config
    import shap

    ax = shap.plots.bar(values, max_display=config.max_display, show=False)
    save_figure(ax.figure, save_fig, config=figure)


def shap_beeswarm(
    values: shap.Explanation,
    save_fig: str | Path,
    *,
    config: ShapConfig | None = None,
    figure: FigureConfig | None = None,
) -> None:
    """Save a SHAP beeswarm plot of per-row feature contributions.

    Parameters
    ----------
    values : shap.Explanation
        Single-output explanation, such as calculate_shap_values returns.
    save_fig : str or pathlib.Path
        Output figure path; parent directories are created.
    config : ShapConfig or None, default None
        Maximum displayed feature count. None uses ShapConfig defaults.
        Sampling settings apply during explanation calculation.

    figure : FigureConfig or None, default None
        Optional export dimensions, DPI, and format. None preserves native
        dimensions and the requested path format.

    Returns
    -------
    None
        The figure is saved to disk; no table is returned.

    Raises
    ------
    ImportError
        The optional shap extra is unavailable.

    Notes
    -----
    Creates parent directories for save_fig and closes the figure after saving.
    Requires ml-toolkit[shap].
    """
    config = ShapConfig() if config is None else config
    import shap

    ax = shap.plots.beeswarm(values, max_display=config.max_display, show=False)
    save_figure(ax.figure, save_fig, config=figure)


def shap_dependence(
    values: shap.Explanation,
    feature: str,
    save_fig: str | Path,
    *,
    figure: FigureConfig | None = None,
) -> None:
    """Save SHAP contribution against one feature, colored by interactions.

    Parameters
    ----------
    values : shap.Explanation
        Single-output explanation containing feature names and data.
    feature : str
        Feature name present in the explanation.
    save_fig : str or pathlib.Path
        Output figure path; parent directories are created.

    figure : FigureConfig or None, default None
        Optional export dimensions, DPI, and format. None preserves native
        dimensions and the requested path format.

    Returns
    -------
    None
        The figure is saved to disk; no table is returned.

    Raises
    ------
    ValueError
        The requested feature is absent from the explanation.
    ImportError
        The optional shap extra is unavailable.

    Notes
    -----
    Creates parent directories for save_fig and closes the figure after saving.
    Requires ml-toolkit[shap].
    """
    if not values.feature_names or feature not in values.feature_names:
        raise ValueError(f"Feature {feature!r} is not in SHAP feature names")
    import shap

    ax = shap.plots.scatter(values[:, feature], color=values, show=False)
    save_figure(ax.figure, save_fig, config=figure)
