"""A project-owned selector that evaluates an approved feature subset."""

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from ml_toolkit import Pool
from ml_toolkit.feature_selection import FeatureSelectionResult, FeatureSelectionStep
from ml_toolkit.metrics import Metric
from ml_toolkit.models.base import Model
from ml_toolkit.types import ModelParams


@dataclass(frozen=True)
class FixedFeatureSelector:
    """Evaluate a nonempty, unique subset, preserving original input order."""

    features: tuple[str, ...]

    def select(
        self,
        *,
        train: Pool,
        validation: Pool,
        params: ModelParams,
        model_factory: Callable[[ModelParams], Model],
        metric: Metric,
    ) -> FeatureSelectionResult:
        """Fit and evaluate the subset; retain no learned state between calls."""
        if not self.features or len(set(self.features)) != len(self.features):
            raise ValueError("features must be nonempty and unique")
        if not set(self.features) <= set(train.X.columns):
            raise ValueError("Requested features must exist in training data")
        names = tuple(name for name in train.X.columns if name in self.features)
        train = train.select_features(names)
        validation = validation.select_features(names)
        started = perf_counter()
        model = model_factory(params.copy()).fit(train, validation)
        fit_seconds = perf_counter() - started
        started = perf_counter()
        prediction = model.predict_pool(validation)
        if prediction.shape != (len(validation),) or not np.isfinite(prediction).all():
            raise ValueError("Predictions must be finite and aligned")
        scores = dict(
            metric(
                validation.y.to_numpy().astype(float),
                prediction,
                validation.weight.to_numpy().astype(float),
            ).to_dict()
        )
        step = FeatureSelectionStep(
            names,
            scores,
            {},
            (),
            fit_seconds,
            perf_counter() - started,
            0.0,
            model.best_iteration,
        )
        return FeatureSelectionResult(
            names, 0, (step,), {"selector": "fixed", "features": names}
        )
