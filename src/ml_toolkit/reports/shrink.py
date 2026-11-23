"""Task-metric curves for independent fitted ensemble prefixes."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from ..data import Pool, validate_split_labels
from ..figures import FigureConfig, PlotError, feature_filename, saved_figure
from ..metrics import Metric, calculate_metrics
from ..models.base import Model, ShrinkableModel
from .base import ModelEvaluator, metric_scores

__all__ = ["ShrinkCurve", "shrink_curve"]


@dataclass(frozen=True, kw_only=True)
class ShrinkCurve(ModelEvaluator):
    """Plot task metrics against the retained boosting iteration count.

    Parameters
    ----------
    name : str, default 'shrink'
        Report name and artifact namespace.
    max_points : int or None, default None
        At least two evenly spaced sizes, including one and the fitted size.
        None uses up to 50 points when no sampling option is supplied.
    step : int or None, default None
        Positive interval: evaluate one, step multiples, and the fitted size.
    sizes : tuple of int or None, default None
        Nonempty, unique positive sizes. The fitted size is always included.
        Sizes must not exceed the actual fitted size, including early stopping.

    Notes
    -----
    Supply at most one sampling option. Models need the optional
    ShrinkableModel capability. Reduced adapters preserve task prediction
    transformations and never refit or change the original model. Every score
    from metric is evaluated per supplied split using row weights. Undefined
    scores become null values and gaps in plots; exceptions propagate.
    write_artifacts calculates once, saving metrics.parquet and one figure per
    metric. evaluate returns no scalar scores. No model size is selected.
    """

    name: str = "shrink"
    max_points: int | None = None
    step: int | None = None
    sizes: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            sum(value is not None for value in (self.max_points, self.step, self.sizes))
            > 1
        ):
            raise ValueError("Specify only one of max_points, step, or sizes")
        if self.max_points is not None and (
            type(self.max_points) is not int or self.max_points < 2
        ):
            raise ValueError("max_points must be an integer of at least two")
        if self.step is not None and (type(self.step) is not int or self.step < 1):
            raise ValueError("step must be a positive integer")
        if self.sizes is not None and (
            type(self.sizes) is not tuple
            or not self.sizes
            or any(type(size) is not int or size < 1 for size in self.sizes)
            or len(set(self.sizes)) != len(self.sizes)
        ):
            raise ValueError(
                "sizes must be a nonempty tuple of unique positive integers"
            )

    def parameters(self) -> dict[str, object]:
        return {
            "max_points": (
                self.max_points
                if self.step is not None or self.sizes is not None
                else self.max_points or 50
            ),
            "step": self.step,
            "sizes": self.sizes,
        }

    def _grid(self, available: int) -> list[int]:
        if type(available) is not int or available < 1:
            raise ValueError("Model n_iterations must be a positive integer")
        if self.sizes is not None:
            if max(self.sizes) > available:
                raise ValueError(
                    f"sizes exceed the fitted model's {available} iterations"
                )
            return sorted({*self.sizes, available})
        if self.step is not None:
            return sorted({1, *range(self.step, available + 1, self.step), available})
        count = min(self.max_points or 50, available)
        return sorted(set(np.linspace(1, available, count, dtype=int).tolist()))

    def calculate(
        self,
        *,
        model: Model | ShrinkableModel,
        metric: Metric,
        data: Pool,
        split: pl.Series,
    ) -> pl.DataFrame:
        """Return n_iterations, split, metric and nullable value in a long table.

        Inputs may contain train only, held-out only, or train plus one of val
        and test. Model capability and requested sizes are checked before any
        reduced predictions. Retain no fitted models or prediction history.
        """
        validate_split_labels(data, split)
        if not len(data):
            raise ValueError("ShrinkCurve requires nonempty data")
        if not isinstance(model, ShrinkableModel):
            raise TypeError(
                f"{type(model).__name__} does not support ShrinkCurve; implement "
                "ShrinkableModel: n_iterations, shrink(n_iterations), and predict_pool(data)"
            )
        available = model.n_iterations
        grid = self._grid(available)
        datasets = [
            (part, data.take(np.flatnonzero((split == part).to_numpy())))
            for part in ("train", "val", "test")
            if (split == part).any()
        ]
        records: list[tuple[int, str, str, float | None]] = []
        names: set[str] | None = None
        for size in grid:
            try:
                current = model if size == available else model.shrink(size)
                for part, pool in datasets:
                    scores = calculate_metrics(metric, pool, current.predict_pool(pool))
                    results = metric_scores(scores)
                    if names is None:
                        names = set(scores)
                    elif names != set(scores):
                        raise ValueError(
                            "Metric names must match across sizes and splits"
                        )
                    records.extend(
                        (size, part, result.name, result.value) for result in results
                    )
                del current
            except Exception as exc:
                exc.add_note(
                    f"Model evaluation evaluator={self.name}, n_iterations={size}"
                )
                raise
        return pl.DataFrame(
            records,
            schema={
                "n_iterations": pl.Int64,
                "split": pl.String,
                "metric": pl.String,
                "value": pl.Float64,
            },
            orient="row",
        )

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
        table = self.calculate(model=model, metric=metric, data=data, split=split)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / "metrics.parquet"
        table.write_parquet(path)
        paths = [path]
        for index, key in enumerate(
            table["metric"].unique(maintain_order=True).to_list()
        ):
            path = artifact_dir / feature_filename(index, key, figure.format)
            shrink_curve(table, metric=key, save_fig=path, figure=figure)
            paths.append(path)
        return paths


def shrink_curve(
    table: pl.DataFrame,
    *,
    metric: str,
    save_fig: str | Path,
    figure: FigureConfig | None = None,
) -> None:
    """Save one metric against retained iterations, with a line per split.

    table is the long table returned by ShrinkCurve.calculate. Null metric
    values leave gaps. Uses natural metric units without assuming a preferred
    direction. The destination's parents are created and figures always close.
    """
    required = {"n_iterations", "split", "metric", "value"}
    if not required <= set(table.columns):
        raise PlotError(
            "Shrink curve table requires n_iterations, split, metric, value"
        )
    selected = table.filter(pl.col("metric") == metric)
    if selected.is_empty():
        raise PlotError(f"No shrink curve values for metric {metric!r}")
    fig, ax = plt.subplots(figsize=(9, 5))
    with saved_figure(fig, save_fig, config=figure):
        for part, label in (
            ("train", "Train"),
            ("val", "Validation"),
            ("test", "Test"),
        ):
            values = selected.filter(pl.col("split") == part).sort("n_iterations")
            if not values.is_empty():
                ax.plot(
                    values["n_iterations"].to_numpy(),
                    values["value"].to_numpy(),
                    marker="o",
                    markersize=3,
                    label=label,
                )
        ax.set(
            xlabel="Retained boosting iterations",
            ylabel=metric,
            title=f"Model shrink: {metric}",
        )
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
