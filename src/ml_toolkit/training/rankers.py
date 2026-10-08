"""Reusable rankers for adapter-driven feature selection."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..config import ShapConfig
from ..explainability import shap_contributions

if TYPE_CHECKING:
    from ..models.base import Model
    from .contracts import TrainingData


def importance_ranker(model: "Model", data: "TrainingData") -> dict[str, float]:
    """Obtain importance values from the adapter's explicit ranking method.

    Parameters
    ----------
    model : Model
        Fitted adapter implementing feature_importances(data).
    data : TrainingData
        Current input columns in fitted order.

    Returns
    -------
    dict[str, float]
        Importance per input feature, using the adapter's backend policy.
        Constant CatBoost trees return zero even when native normalization
        produces NaN.

    Notes
    -----
    The adapter owns importance calculation and feature-order validation.
    This ranker does not discover backend methods or invoke feature selection.
    """
    return model.feature_importances(data)


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

    def __post_init__(self) -> None:
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
            weights.shape != (len(data),)
            or not np.isfinite(weights).all()
            or (weights < 0).any()
            or not (weights > 0).any()
        ):
            raise ValueError("SHAP ranking requires valid weights with positive total")
        eligible = np.flatnonzero(weights > 0)
        if len(eligible) > self.max_samples:
            eligible = np.sort(
                np.random.default_rng(self.seed).choice(
                    eligible,
                    self.max_samples,
                    replace=False,
                )
            )
        sampled = data.take(eligible)
        names, _, contributions = shap_contributions(
            model.native_model,
            sampled.X,
            sampled.cat_features,
        )
        if len(names) != len(data.X.columns) or set(names) != set(data.X.columns):
            raise ValueError("SHAP feature names must match the current input columns")
        values = np.average(
            np.abs(contributions[:, :-1]),
            axis=0,
            weights=weights[eligible],
        )
        if not np.isfinite(values).all():
            raise ValueError("SHAP importances must be finite")
        return dict(zip(names, values.tolist(), strict=True))
