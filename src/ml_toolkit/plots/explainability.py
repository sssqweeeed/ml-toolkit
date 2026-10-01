"""Model-agnostic feature importance and optional SHAP visualizations."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..explainability import top_shap_features
from .utils import numeric_vector, same_length, save_figure


def feature_importance(feature_names, importance, save_fig: str | Path, *,
                       max_features_display: int = 10) -> pd.DataFrame:
    """Plot supplied importance values without requiring a model-specific object."""
    values = numeric_vector(importance, "importance")
    names = list(feature_names)
    same_length(feature_names=names, importance=values)
    if max_features_display < 1:
        raise ValueError("max_features_display must be positive")
    ordered = (pd.DataFrame({"feature": names, "importance": values})
               .sort_values("importance", ascending=False, kind="stable")
               .head(max_features_display))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(ordered.feature.iloc[::-1], ordered.importance.iloc[::-1])
    ax.set(xlabel="Feature importance", ylabel="Feature", title="Feature importance")
    fig.tight_layout()
    save_figure(fig, save_fig)
    return ordered.reset_index(drop=True)


def _shap():
    try:
        import shap
    except ImportError as error:
        raise ImportError("Install ml-toolkit[shap] to use SHAP plots") from error
    return shap


def shap_importance(values, save_fig: str | Path, *, max_features_display: int = 10) -> None:
    shap = _shap()
    ax = shap.plots.bar(values, max_display=max_features_display, show=False)
    save_figure(ax.figure, save_fig)


def shap_beeswarm(values, save_fig: str | Path, *, max_features_display: int = 10) -> None:
    shap = _shap()
    ax = shap.plots.beeswarm(values, max_display=max_features_display, show=False)
    save_figure(ax.figure, save_fig)


def shap_dependence(values, feature: str, save_fig: str | Path) -> None:
    if not values.feature_names or feature not in values.feature_names:
        raise ValueError(f"Feature {feature!r} is not in SHAP feature names")
    shap = _shap()
    ax = shap.plots.scatter(values[:, feature], color=values, show=False)
    save_figure(ax.figure, save_fig)
