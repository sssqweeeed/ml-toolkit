"""Reusable rankers for adapter-driven feature selection."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import catboost
import numpy as np

from ..config import ShapConfig
from ..explainability import _shap_contributions

if TYPE_CHECKING:
    from ..models.base import Model
    from .contracts import TrainingData


def _constant_tree_outputs(native):
    counts = native.get_tree_leaf_counts()
    values = native.get_leaf_values()
    total = int(np.sum(counts))
    if not total or values.size % total or not np.isfinite(values).all():
        return False
    leaves = values.reshape(total, -1)
    offset = 0
    for count in counts:
        tree = leaves[offset : offset + count]
        if not (tree == tree[0]).all():
            return False
        offset += count
    return True


def importance_ranker(model: "Model", data: "TrainingData") -> dict[str, float]:
    """Rank an adapter's inputs by fitted native importances or coefficients.

    Parameters
    ----------
    model : Model
        Fitted adapter exposing native_model. CatBoost uses PredictionValuesChange;
        LightGBM Boosters use gain; other estimators expose feature_importances_
        or coef_. This function never invokes a native feature selector.
    data : TrainingData
        Current input columns. Names must match the fitted native feature names.

    Returns
    -------
    dict[str, float]
        Importance per input name. Coefficients use absolute values, averaged
        across outputs when two-dimensional. Constant CatBoost tree outputs
        have zero importance, including when native normalization returns NaN.

    Raises
    ------
    ValueError
        Native feature names or importance dimensions do not match the inputs.
        Importance values are nonfinite.
        The estimator exposes no supported importance or coefficient API.
    """
    native = model.native_model
    if isinstance(native, catboost.CatBoost):
        names = list(native.feature_names_)
        values = np.asarray(
            native.get_feature_importance(type="PredictionValuesChange"), dtype=float
        )
        if not np.isfinite(values).all() and _constant_tree_outputs(native):
            values = np.zeros(len(names))
    elif hasattr(native, "feature_name") and hasattr(native, "feature_importance"):
        names = list(native.feature_name())
        values = native.feature_importance(importance_type="gain")
    else:
        names = list(getattr(native, "feature_names_in_", data.X.columns))
        if hasattr(native, "feature_importances_"):
            values = native.feature_importances_
        elif hasattr(native, "coef_"):
            values = np.abs(np.asarray(native.coef_, dtype=float))
            if values.ndim == 2:
                values = values.mean(axis=0)
        else:
            raise ValueError("Native model exposes no feature importances or coefficients")
    values = np.asarray(values, dtype=float)
    if len(names) != len(data.X.columns) or set(names) != set(data.X.columns):
        raise ValueError("Native feature names must match the current input columns")
    if values.shape != (len(names),):
        raise ValueError("Native importance dimensions must match the input columns")
    if not np.isfinite(values).all():
        raise ValueError("Native importances must be finite")
    return dict(zip(names, values.tolist(), strict=True))


@dataclass(frozen=True, kw_only=True)
class ShapRanker:
    """Rank native tree features by weighted mean absolute raw-output SHAP.

    Parameters
    ----------
    max_samples : int, default 1000
        Positive maximum sampled validation rows. Samples uniformly without
        replacement among positive-weight rows, then aggregates using weights.
    seed : int, default 0
        Nonnegative sampling seed. Fixed split rows give the same sample each
        step, independent of the selected feature names.

    Notes
    -----
    Supports native CatBoost and LightGBM Boosters exposed by model.native_model.
    Handles CatBoost null categories using the shared unknown-category policy.
    Explains the inner raw-output model, including for inflation adapters;
    RFE evaluates the complete adapter separately on its task metric. Uses
    backend SHAP contributions directly and does not require the shap extra.
    """

    max_samples: int = 1000
    seed: int = 0

    def __post_init__(self):
        ShapConfig(max_samples=self.max_samples, seed=self.seed)

    def __call__(self, model: "Model", data: "TrainingData") -> dict[str, float]:
        """Return weighted SHAP importances for the current feature subset.

        Parameters
        ----------
        model : Model
            Fitted adapter exposing a supported single-output native tree model.
        data : TrainingData
            Validation rows with finite nonnegative weights and positive total.

        Returns
        -------
        dict[str, float]
            Weighted mean absolute SHAP contribution per fitted feature.

        Raises
        ------
        ValueError
            Weights, native feature names or SHAP dimensions are invalid.
        TypeError
            The native backend is unsupported.
        ImportError
            A working LightGBM installation is unavailable for that backend.
        """
        weights = np.asarray(data.weight.to_numpy(), dtype=float)
        if (
            weights.shape != (len(data),) or not np.isfinite(weights).all()
            or (weights < 0).any() or not (weights > 0).any()
        ):
            raise ValueError("SHAP ranking requires valid weights with positive total")
        eligible = np.flatnonzero(weights > 0)
        if len(eligible) > self.max_samples:
            eligible = np.sort(np.random.default_rng(self.seed).choice(
                eligible, self.max_samples, replace=False,
            ))
        sampled = data.take(eligible)
        names, _, contributions = _shap_contributions(
            model.native_model, sampled.X, sampled.cat_features,
        )
        if len(names) != len(data.X.columns) or set(names) != set(data.X.columns):
            raise ValueError("SHAP feature names must match the current input columns")
        values = np.average(
            np.abs(contributions[:, :-1]), axis=0, weights=weights[eligible],
        )
        if not np.isfinite(values).all():
            raise ValueError("SHAP importances must be finite")
        return dict(zip(names, values.tolist(), strict=True))
