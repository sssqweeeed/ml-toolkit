"""A project-owned permutation ranker using weighted validation MAE."""

from dataclasses import dataclass, replace

import numpy as np

from ml_toolkit.data import Pool
from ml_toolkit.models import Model


@dataclass(frozen=True)
class PermutationMAERanker:
    """Rank columns by the rise in weighted MAE after one seeded permutation.

    Higher scores identify columns whose disruption hurts predictions more.
    Values may be negative. The model remains fitted; only a copy of one feature
    column changes for each prediction. Targets, weights, timestamps and row
    metadata stay in their original order. No predictions or models are stored.
    """

    seed: int = 42

    def __call__(self, model: Model, data: Pool) -> dict[str, float]:
        """Return one finite importance for every current validation column."""
        target = data.y.to_numpy()
        weights = data.weight.to_numpy()
        baseline = float(
            np.average(np.abs(target - model.predict_pool(data)), weights=weights)
        )
        rng = np.random.default_rng(self.seed)
        importance: dict[str, float] = {}
        for name in data.X.columns:
            shuffled = data.X[name].gather(rng.permutation(len(data)))
            perturbed = replace(data, X=data.X.with_columns(shuffled))
            error = float(
                np.average(
                    np.abs(target - model.predict_pool(perturbed)), weights=weights
                )
            )
            importance[name] = error - baseline
        return importance
