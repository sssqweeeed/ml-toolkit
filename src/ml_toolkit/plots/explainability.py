"""Model-agnostic feature importance and optional SHAP visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
from numpy.typing import ArrayLike

from ..config import FigureConfig, ShapConfig
from .utils import numeric_vector, same_length, save_figure

if TYPE_CHECKING:
    import shap


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
    ax.barh(ordered.feature.iloc[::-1], ordered.importance.iloc[::-1])
    ax.set(xlabel="Feature importance", ylabel="Feature", title="Feature importance")
    fig.tight_layout()
    save_figure(fig, save_fig, config=figure)
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
