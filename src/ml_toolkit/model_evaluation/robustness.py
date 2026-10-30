"""Task-metric curves for explicit changes to held-out input features."""

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
from ..plots.utils import feature_filename
from ..types import FloatArray

_SCHEMA = {
    "split": pl.String,
    "mode": pl.String,
    "level": pl.Float64,
    "repeat": pl.Int64,
    "is_baseline": pl.Boolean,
    "metric": pl.String,
    "value": pl.Float64,
    "delta": pl.Float64,
    "n_rows": pl.Int64,
    "sum_weight": pl.Float64,
}


@dataclass(frozen=True, kw_only=True)
class RobustnessCurve(ModelEvaluator):
    """Measure task metrics after controlled changes to held-out features.

    Parameters
    ----------
    features : tuple of str
        Nonempty, unique feature names from Pool.X, changed together. Auxiliary
        columns are never passed to the model or changed by this report.
    mode : {'missing', 'noise', 'rounding', 'unknown'}, default 'missing'
        Missing injects nulls. Unknown injects unknown_value only in categorical
        model columns, converting those changed columns to String locally.
        Noise adds independent Gaussian noise to numerical features, scaled by
        each feature's weighted population standard deviation on train rows.
        Rounding rounds numerical features to the requested decimal places.
    levels : tuple of float, default (0.05, 0.1, 0.2, 0.4)
        Unique, finite scenario levels. Missing/unknown use fractions in [0, 1].
        Noise uses nonnegative standard-deviation multipliers. Rounding uses
        nonnegative integer decimal places, with smaller values coarser.
    n_repeats : int, default 3
        Positive number of random realizations per level. Rounding is
        deterministic and is calculated once per level.
    max_samples : int or None, default 10000
        Positive maximum number of uniformly sampled positive-weight held-out
        rows. Zero-weight rows are excluded before sampling. Sampling occurs
        once and retains the same rows, targets and weights at all levels.
        None evaluates all positive-weight held-out rows.
    seed : int, default 0
        Nonnegative seed for row sampling and perturbations. Each realization
        shares masks/noise across levels so fractions form nested row sets.
    unknown_value : str, default 'unknown'
        Unseen label used by unknown mode. Must not occur in any supplied value
        of a selected feature. The same label is used for all selected features.
    name : str, default 'robustness'
        Unique report name and artifact namespace.

    Notes
    -----
    Evaluates the fitted adapter without refitting or mutating inputs. Requires
    val or test rows; train-only inputs return an empty table and no artifacts.
    Noise additionally requires train rows to estimate its scale, excluding
    zero-weight, null and nonfinite feature values. A constant feature has zero
    noise scale. Changed numerical columns may be promoted to Float64.

    Every named score from the supplied task metric is evaluated with the
    original held-out targets and row weights. Nonfinite scores become nulls.
    The unmodified baseline is a separate row with level/repeat=null and
    is_baseline=True; delta is changed score minus baseline in natural units.
    A positive delta improves a maximized metric and worsens a minimized one.
    """

    features: tuple[str, ...]
    mode: Literal["missing", "noise", "rounding", "unknown"] = "missing"
    levels: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4)
    n_repeats: int = 3
    max_samples: int | None = 10_000
    seed: int = 0
    unknown_value: str = "unknown"
    name: str = "robustness"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.features or any(not name.strip() for name in self.features):
            raise ValueError("features must contain nonempty feature names")
        if len(set(self.features)) != len(self.features):
            raise ValueError("features must contain unique feature names")
        if self.mode not in ("missing", "noise", "rounding", "unknown"):
            raise ValueError("Unsupported robustness mode")
        if not self.levels or any(not np.isfinite(level) for level in self.levels):
            raise ValueError("levels must be nonempty and finite")
        if len(set(self.levels)) != len(self.levels):
            raise ValueError("levels must be unique")
        if any(level < 0 for level in self.levels):
            raise ValueError("levels must be nonnegative")
        if self.mode in ("missing", "unknown") and any(
            level > 1 for level in self.levels
        ):
            raise ValueError("Missing/unknown levels must be fractions in [0, 1]")
        if self.mode == "rounding" and any(
            level != int(level) for level in self.levels
        ):
            raise ValueError("Rounding levels must be integer decimal places")
        if self.n_repeats < 1 or self.n_repeats != int(self.n_repeats):
            raise ValueError("n_repeats must be a positive integer")
        if self.max_samples is not None and (
            self.max_samples < 1 or self.max_samples != int(self.max_samples)
        ):
            raise ValueError("max_samples must be a positive integer or None")
        if self.seed < 0 or self.seed != int(self.seed):
            raise ValueError("seed must be a nonnegative integer")
        if not self.unknown_value:
            raise ValueError("unknown_value must be nonempty")

    def parameters(self) -> dict[str, object]:
        return {
            "features": self.features,
            "mode": self.mode,
            "levels": self.levels,
            "n_repeats": self.n_repeats,
            "max_samples": self.max_samples,
            "seed": self.seed,
            "unknown_value": self.unknown_value,
        }

    def _validate_features(self, data: Pool) -> None:
        missing = sorted(set(self.features) - set(data.X.columns))
        if missing:
            raise ValueError(f"Robustness features are missing from X: {missing}")
        categorical = set(data.cat_features)
        if self.mode == "unknown" and not set(self.features) <= categorical:
            raise ValueError(
                "Unknown-category scenarios require categorical model features"
            )
        if self.mode == "unknown" and any(
            data.X[name].cast(pl.String).eq(self.unknown_value).any()
            for name in self.features
        ):
            raise ValueError("unknown_value must not occur in selected feature values")
        if self.mode in ("noise", "rounding") and any(
            name in categorical or not data.X.schema[name].is_numeric()
            for name in self.features
        ):
            raise ValueError(
                "Noise/rounding scenarios require numerical model features"
            )

    def _noise_scales(self, data: Pool, split: pl.Series) -> dict[str, float]:
        index = np.flatnonzero((split == "train").to_numpy())
        if not len(index):
            raise ValueError(
                "Noise scenarios require train rows to estimate feature scales"
            )
        weight = data.weight.gather(index).to_numpy().astype(float)
        scales: dict[str, float] = {}
        for name in self.features:
            values = data.X[name].gather(index).cast(pl.Float64).to_numpy()
            selected = np.isfinite(values) & (weight > 0)
            if not selected.any():
                raise ValueError(
                    f"No finite positive-weight training values for {name!r}"
                )
            selected_values, selected_weight = values[selected], weight[selected]
            center = np.average(selected_values, weights=selected_weight)
            scales[name] = float(
                np.sqrt(
                    np.average((selected_values - center) ** 2, weights=selected_weight)
                )
            )
            if not np.isfinite(scales[name]):
                raise ValueError(
                    f"Training standard deviation must be finite for {name!r}"
                )
        return scales

    def _changed_pool(
        self,
        data: Pool,
        level: float,
        disturbance: dict[str, FloatArray],
        scales: dict[str, float],
    ) -> Pool:
        columns: list[pl.Series] = []
        for name in self.features:
            column = data.X[name]
            if self.mode == "missing":
                changed = column.set(pl.Series(disturbance[name] < level), None)
            elif self.mode == "unknown":
                changed = column.cast(pl.String).set(
                    pl.Series(disturbance[name] < level), self.unknown_value
                )
            elif self.mode == "noise":
                values = column.cast(pl.Float64).to_numpy()
                changed = pl.Series(
                    name, values + disturbance[name] * level * scales[name]
                )
                changed = changed.set(column.is_null(), None)
            else:
                changed = column.round(decimals=int(level))
            columns.append(changed)
        return replace(data, X=data.X.with_columns(columns))

    def calculate(
        self,
        *,
        model: Model | ShrinkableModel,
        metric: Metric,
        data: Pool,
        split: pl.Series,
    ) -> pl.DataFrame:
        """Return baseline and repeated scenario scores in a typed long table.

        Parameters
        ----------
        model : fitted prediction adapter
            Supplies predict_pool and preserves its own learned preprocessing.
        metric : Metric
            Task callback accepting target, prediction and original row weights.
        data : Pool
            Aligned train and optional single held-out sample, or held-out only.
        split : polars.Series
            Aligned non-null String labels containing train, val or test.

        Returns
        -------
        polars.DataFrame
            Columns split, mode, nullable level/repeat, is_baseline, metric,
            nullable value/delta, n_rows, and sum_weight. A baseline uses the
            same sampled positive-weight held-out rows as every scenario.
            n_rows counts those rows; zero-weight rows are excluded. Empty for
            train-only. All-zero held-out weights raise ValueError.
        """
        validate_split(data, split)
        index = np.flatnonzero((split != "train").to_numpy())
        if not len(index):
            return pl.DataFrame(schema=_SCHEMA)
        self._validate_features(data)
        target = data.y.to_numpy().astype(float)
        weight = data.weight.to_numpy().astype(float)
        if not np.isfinite(target).all():
            raise ValueError("Target values must be finite")
        if not np.isfinite(weight).all() or (weight < 0).any():
            raise ValueError("Weights must be finite and nonnegative")
        index = index[weight[index] > 0]
        if not len(index):
            raise ValueError("Held-out rows must have positive total weight")
        rng = np.random.default_rng(self.seed)
        if self.max_samples is not None and len(index) > self.max_samples:
            index = np.sort(rng.choice(index, size=self.max_samples, replace=False))
        part = str(split[int(index[0])])
        heldout = data.take(index)
        total_weight = float(heldout.weight.sum())
        if total_weight <= 0:
            raise ValueError("Sampled held-out rows must have positive total weight")
        scales = self._noise_scales(data, split) if self.mode == "noise" else {}
        try:
            baseline = calculate_metrics(metric, heldout, model.predict_pool(heldout))
        except Exception as exc:
            exc.add_note(
                f"Model evaluation evaluator={self.name}, mode={self.mode}, level=baseline, repeat=baseline"
            )
            raise
        if not baseline or any(not name.strip() for name in baseline):
            raise ValueError("Metric must return nonempty named scores")
        names = set(baseline)
        records: list[
            tuple[
                str,
                str,
                float | None,
                int | None,
                bool,
                str,
                float | None,
                float | None,
                int,
                float,
            ]
        ] = []
        for name, score in baseline.items():
            value = float(score)
            records.append(
                (
                    part,
                    self.mode,
                    None,
                    None,
                    True,
                    name,
                    value if np.isfinite(value) else None,
                    0.0 if np.isfinite(value) else None,
                    len(heldout),
                    total_weight,
                )
            )
        repeats = 1 if self.mode == "rounding" else self.n_repeats
        for repeat in range(repeats):
            generator = np.random.default_rng(
                np.random.SeedSequence([self.seed, repeat, 1])
            )
            disturbance = {
                name: (
                    generator.standard_normal(len(heldout))
                    if self.mode == "noise"
                    else generator.random(len(heldout))
                )
                for name in self.features
            }
            for level in sorted(self.levels):
                try:
                    changed = self._changed_pool(heldout, level, disturbance, scales)
                    scores = calculate_metrics(
                        metric, changed, model.predict_pool(changed)
                    )
                    if set(scores) != names:
                        raise ValueError(
                            "Metric names must match across baseline and scenarios"
                        )
                    for name, score in scores.items():
                        value, original = float(score), float(baseline[name])
                        delta = value - original
                        records.append(
                            (
                                part,
                                self.mode,
                                float(level),
                                repeat,
                                False,
                                name,
                                value if np.isfinite(value) else None,
                                delta if np.isfinite(delta) else None,
                                len(heldout),
                                total_weight,
                            )
                        )
                except Exception as exc:
                    exc.add_note(
                        f"Model evaluation evaluator={self.name}, mode={self.mode}, level={level}, repeat={repeat}"
                    )
                    raise
        return pl.DataFrame(records, schema=_SCHEMA, orient="row")

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
        from ..plots.robustness import robustness_curve

        table = self.calculate(model=model, metric=metric, data=data, split=split)
        if table.is_empty():
            return []
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / "metrics.parquet"
        table.write_parquet(path)
        paths = [path]
        for index, name in enumerate(
            table["metric"].unique(maintain_order=True).to_list()
        ):
            path = artifact_dir / feature_filename(index, name, figure.format)
            robustness_curve(table, metric=name, save_fig=path, figure=figure)
            paths.append(path)
        return paths
