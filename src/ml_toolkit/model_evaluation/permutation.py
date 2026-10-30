"""Held-out task-score changes under repeated feature permutations."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

from .base import ModelEvaluator
from ..config import FigureConfig
from ..data import Pool
from ..data_analysis._common import validate_split
from ..metrics import Metric
from ..metrics._evaluation import calculate_metrics
from ..models.base import Model, ShrinkableModel


@dataclass(frozen=True, kw_only=True)
class PermutationImportance(ModelEvaluator):
    """Measure held-out score deterioration after shuffling features or groups.

    Parameters
    ----------
    score : str
        Named score returned by the task metric callback.
    direction : {'minimize', 'maximize'}
        Preferred direction of score. Positive importance always means that
        shuffling worsened the score; negative importance means it improved.
    features : tuple of str or None, default None
        Individual model features to shuffle. None selects all columns not
        assigned to groups; () selects no individual columns.
    groups : tuple of tuples of str, default ()
        Disjoint nonempty feature groups, shuffled together with one common
        row permutation. Columns cannot also occur in features.
    n_repeats : int, default 5
        Positive number of independent permutations per individual/group.
    max_samples : int, default 10000
        Positive maximum positive-weight held-out row count, sampled uniformly without
        replacement. The same rows and row weights are used for every score.
    seed : int, default 0
        Nonnegative random seed for sampling and permutations.
    name : str, default 'permutation_importance'
        Report name and artifact namespace.

    Notes
    -----
    Uses validation or test rows only and skips train-only fits. No fitting or
    mutation takes place. This measures reliance of a fixed model on selected
    features; correlated alternatives can mask individual importance. Group
    related features when their joint contribution is the relevant question.
    The graph shows repeat mean and standard deviation, not a confidence
    interval. Nonfinite metric values are null; metric exceptions propagate.
    """

    score: str
    direction: Literal["minimize", "maximize"]
    features: tuple[str, ...] | None = None
    groups: tuple[tuple[str, ...], ...] = ()
    n_repeats: int = 5
    max_samples: int = 10_000
    seed: int = 0
    name: str = "permutation_importance"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.score.strip():
            raise ValueError("score must not be empty")
        if self.direction not in ("minimize", "maximize"):
            raise ValueError("direction must be minimize or maximize")
        if self.n_repeats < 1 or self.max_samples < 1:
            raise ValueError("n_repeats and max_samples must be positive")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        single = () if self.features is None else self.features
        grouped = tuple(feature for group in self.groups for feature in group)
        all_names = (*single, *grouped)
        if any(not group for group in self.groups):
            raise ValueError("Feature groups must not be empty")
        if any(not feature.strip() for feature in all_names):
            raise ValueError("Feature names must not be empty")
        if len(set(all_names)) != len(all_names):
            raise ValueError(
                "Individual features and groups must be disjoint and unique"
            )
        if self.features == () and not self.groups:
            raise ValueError("Select at least one individual feature or group")

    def parameters(self) -> dict[str, object]:
        return {
            "score": self.score,
            "direction": self.direction,
            "features": self.features,
            "groups": self.groups,
            "n_repeats": self.n_repeats,
            "max_samples": self.max_samples,
            "seed": self.seed,
        }

    def calculate(
        self,
        *,
        model: Model | ShrinkableModel,
        metric: Metric,
        data: Pool,
        split: pl.Series,
    ) -> pl.DataFrame:
        """Return per-repeat baseline, shuffled score and oriented importance.

        Returns
        -------
        polars.DataFrame
            Columns split, group_id, features (List[String]), repeat (zero-based),
            metric, baseline, value, importance, n_rows and sum_weight. Scores
            and importance are nullable Float64. Rows represent individual
            permutations; an empty typed table indicates no held-out rows.
        """
        schema = {
            "split": pl.String,
            "group_id": pl.String,
            "features": pl.List(pl.String),
            "repeat": pl.Int64,
            "metric": pl.String,
            "baseline": pl.Float64,
            "value": pl.Float64,
            "importance": pl.Float64,
            "n_rows": pl.Int64,
            "sum_weight": pl.Float64,
        }
        validate_split(data, split)
        heldout = np.flatnonzero((split != "train").to_numpy())
        if not len(heldout):
            return pl.DataFrame(schema=schema)
        target = data.y.gather(heldout).to_numpy().astype(float)
        weight = data.weight.gather(heldout).to_numpy().astype(float)
        if not np.isfinite(target).all():
            raise ValueError("Targets must be finite")
        if (
            not np.isfinite(weight).all()
            or (weight < 0).any()
            or not (weight > 0).any()
        ):
            raise ValueError(
                "Weights must be finite, nonnegative and have positive total"
            )
        heldout = heldout[weight > 0]
        grouped = {feature for group in self.groups for feature in group}
        single = (
            tuple(feature for feature in data.X.columns if feature not in grouped)
            if self.features is None
            else self.features
        )
        groups = (*tuple((feature,) for feature in single), *self.groups)
        missing = {feature for group in groups for feature in group} - set(
            data.X.columns
        )
        if missing:
            raise ValueError(
                f"Permutation features are missing from X: {sorted(missing)}"
            )
        if not groups:
            raise ValueError("Select at least one individual feature or group")
        rng = np.random.default_rng(self.seed)
        if len(heldout) > self.max_samples:
            heldout = np.sort(rng.choice(heldout, self.max_samples, replace=False))
        pool = data.take(heldout)
        part = str(split[int(heldout[0])])
        weight = pool.weight.to_numpy().astype(float)
        baseline = self._score(model, metric, pool)
        records: list[dict[str, object]] = []
        for group_index, features in enumerate(groups):
            for repeat in range(self.n_repeats):
                try:
                    permutation = rng.permutation(len(pool))
                    frame = pool.X.with_columns(
                        [pool.X[feature].gather(permutation) for feature in features]
                    )
                    value = self._score(model, metric, replace(pool, X=frame))
                    delta = (
                        None
                        if value is None or baseline is None
                        else (
                            value - baseline
                            if self.direction == "minimize"
                            else baseline - value
                        )
                    )
                    if delta is not None and not np.isfinite(delta):
                        delta = None
                    records.append(
                        {
                            "split": part,
                            "group_id": f"g{group_index:03d}",
                            "features": list(features),
                            "repeat": repeat,
                            "metric": self.score,
                            "baseline": baseline,
                            "value": value,
                            "importance": delta,
                            "n_rows": len(pool),
                            "sum_weight": float(weight.sum()),
                        }
                    )
                except Exception as exc:
                    exc.add_note(
                        f"Permutation evaluation evaluator={self.name}, features={features}, repeat={repeat}"
                    )
                    raise
        return pl.DataFrame(records, schema=schema)

    def _score(
        self, model: Model | ShrinkableModel, metric: Metric, data: Pool
    ) -> float | None:
        values = calculate_metrics(metric, data, model.predict_pool(data))
        if self.score not in values:
            raise ValueError(f"Task metric did not return score {self.score!r}")
        value = float(values[self.score])
        return value if np.isfinite(value) else None

    def write_artifacts(
        self,
        *,
        model: Model | ShrinkableModel,
        metric: Metric,
        data: Pool,
        split: pl.Series,
        artifact_dir: Path,
        figure: FigureConfig,
    ) -> list[Path]:
        from ..plots.permutation_importance import permutation_importance

        table = self.calculate(model=model, metric=metric, data=data, split=split)
        if table.is_empty():
            return []
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / "importance.parquet"
        table.write_parquet(path)
        image = artifact_dir / f"importance.{figure.format}"
        permutation_importance(table, save_fig=image, figure=figure)
        return [path, image]
