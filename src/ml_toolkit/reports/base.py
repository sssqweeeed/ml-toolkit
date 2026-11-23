"""Report base classes and the input, name and metric helpers reports share."""

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from ..data import Pool, row_indexer, validate_prediction, validate_split_labels
from ..feature_levels import FeatureLevelConfig
from ..figures import FigureConfig
from ..metrics import Metric
from ..models.base import Model, ShrinkableModel
from ..types import FloatArray

__all__ = [
    "DataAnalyzer",
    "MetricScore",
    "ModelEvaluator",
    "PredictionEvaluator",
    "comparison_pair",
    "data_comparison_pair",
    "expand_skipped_rows",
    "long_metric_rows",
    "metric_scores",
    "validate_inputs",
    "validate_report_metrics",
    "validate_report_name",
    "write_heldout_plot",
]


def validate_report_name(name: str, kind: str) -> None:
    """Require a name usable as an artifact namespace and metric prefix."""
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) is None:
        raise ValueError(
            f"{kind} name must start with an ASCII letter or digit and contain only ASCII letters, digits, underscores, hyphens or dots"
        )


@dataclass(frozen=True, kw_only=True)
class PredictionEvaluator:
    """Named additional metrics and artifacts; neither method is mandatory.

    ``evaluate`` keys are logged under ``<name>/<key>`` and do not become CV
    averages or ordinary training results. Pass this instance to
    ``TuningConfig.evaluator`` to select one returned key as a trial objective.
    ``RFESelector.evaluator`` and ``ForwardSelector.evaluator`` select candidate
    feature subsets in the same way, retaining scores in selection history.
    Return finite scalar values only.
    ``write_artifacts`` receives shared figure and feature-level settings.
    Only returned files are uploaded under ``<name>/``. The directory is deleted
    after uploading.
    """

    name: str

    def __post_init__(self) -> None:
        validate_report_name(self.name, "Evaluator")

    def evaluate(
        self, *, data: Pool, prediction: FloatArray, split: pl.Series
    ) -> dict[str, float]:
        """Return additional scalar metrics, or an empty dictionary.

        Parameters
        ----------
        data : Pool
            Combined train and optional held-out rows. Do not mutate its fields.
        prediction : FloatArray
            Finite one-dimensional predictions aligned with data, shape (n_rows,).
        split : polars.Series
            Aligned non-null String labels: train plus at most one of val/test.

        Returns
        -------
        dict of str to float
            Finite values logged as ``<name>/<key>``. Include the sample in keys
            when relevant. Evaluator-based tuning, RFE, and Forward select an exact
            returned key through their metric setting. Calculate that score from
            validation rows, identified by split == "val". Standalone selection
            returns scores in its history without logging them.
        """
        return {}

    def write_artifacts(
        self,
        *,
        data: Pool,
        prediction: FloatArray,
        split: pl.Series,
        artifact_dir: Path,
        figure: FigureConfig,
        features: tuple[FeatureLevelConfig, ...] | None,
        n_bins: int,
    ) -> list[Path]:
        """Write report files and return only the paths to upload.

        Parameters
        ----------
        data : Pool
            Combined train and optional held-out rows; inputs must not be mutated.
        prediction : FloatArray
            Finite predictions aligned with data, shape (n_rows,).
        split : polars.Series
            Aligned non-null String labels: train plus at most one of val/test.
        artifact_dir : pathlib.Path
            Temporary directory owned by the caller, removed after upload.
        figure : FigureConfig
            Shared figure dimensions, resolution and file format.
        features : tuple of FeatureLevelConfig or None
            Shared feature-level rules. None infers rules; an empty tuple disables
            level diagnostics. Use only if relevant to this report.
        n_bins : int
            Shared numerical feature-level bin limit.

        Returns
        -------
        list of pathlib.Path
            Files relative to artifact_dir, or absolute paths inside it. Only
            these files are uploaded under the report name. Return [] to skip.
        """
        return []

    def parameters(self) -> dict[str, object]:
        """Return an explicit JSON-compatible settings description, never state."""
        return {}


@dataclass(frozen=True, kw_only=True)
class ModelEvaluator:
    """Named model report with optional scalar metrics and selected artifacts.

    Both methods receive the fitted adapter, the training metric callback and
    aligned data/split labels. Do not mutate inputs or retain fitted state:
    evaluator instances are reused across folds. Scalar results are additional
    reports, never training objectives or validation averages. Only returned
    artifact paths inside artifact_dir are uploaded beneath name/.
    """

    name: str

    def __post_init__(self) -> None:
        validate_report_name(self.name, "Evaluator")

    def evaluate(
        self,
        *,
        model: Model | ShrinkableModel,
        metric: Metric,
        data: Pool,
        split: pl.Series,
    ) -> dict[str, float]:
        """Return finite additional scalar scores, or {} to omit them."""
        return {}

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
        """Write report files and return relative or contained absolute paths."""
        return []

    def parameters(self) -> dict[str, object]:
        """Return JSON-compatible settings, excluding fitted state."""
        return {}


@dataclass(frozen=True, kw_only=True)
class DataAnalyzer:
    """Named data metrics and artifacts, independent of model predictions.

    Implement either hook or both. During training, finite metrics are logged
    under ``<name>/<key>`` and only returned files are uploaded under ``<name>/``.
    These metrics do not become CV averages or optimization objectives.
    Names must be unique across analyzers and prediction evaluators.
    """

    name: str

    def __post_init__(self) -> None:
        validate_report_name(self.name, "Analyzer")

    def evaluate(self, *, data: Pool, split: pl.Series) -> dict[str, float]:
        """Return finite scalar measurements, or an empty dictionary.

        Parameters
        ----------
        data : Pool
            Combined train and optional held-out rows. Do not mutate its fields.
        split : polars.Series
            Aligned non-null String labels: train plus at most one of val/test.

        Returns
        -------
        dict of str to float
            Additional values logged as ``<name>/<key>``. Include sample names
            in keys where relevant.
        """
        return {}

    def write_artifacts(
        self,
        *,
        data: Pool,
        split: pl.Series,
        artifact_dir: Path,
        figure: FigureConfig,
        features: tuple[FeatureLevelConfig, ...] | None,
        n_bins: int,
    ) -> list[Path]:
        """Write data reports and return only files to upload.

        Parameters
        ----------
        data : Pool
            Combined train and optional held-out rows; inputs must not be mutated.
        split : polars.Series
            Aligned non-null String labels: train plus at most one of val/test.
        artifact_dir : pathlib.Path
            Temporary directory owned by the caller, removed after uploading.
        figure : FigureConfig
            Shared figure dimensions, resolution and format.
        features : tuple of FeatureLevelConfig or None
            Shared feature-level rules; None infers rules and () disables levels.
            Use only if relevant to this analysis.
        n_bins : int
            Shared numerical feature-level bin limit.

        Returns
        -------
        list of pathlib.Path
            Files relative to artifact_dir, or absolute paths inside it.
            Only these files are uploaded under the analyzer name.
        """
        return []

    def parameters(self) -> dict[str, object]:
        """Return explicit JSON-compatible settings, never data or fitted state."""
        return {}


def validate_inputs(data: Pool, prediction: FloatArray, split: pl.Series) -> None:
    """Require predictions and split labels aligned with the pool rows."""
    validate_prediction(data, prediction)
    validate_split_labels(data, split)


def _heldout(
    data: Pool, prediction: FloatArray, split: pl.Series
) -> tuple[str, Pool, FloatArray] | None:
    validate_inputs(data, prediction, split)
    index = np.flatnonzero((split != "train").to_numpy())
    if not len(index):
        return None
    part = str(split[int(index[0])])
    return part, data.take(index), prediction[row_indexer(index, len(data))]


def comparison_pair(
    data: Pool, prediction: FloatArray, split: pl.Series
) -> tuple[str, Pool, FloatArray, Pool, FloatArray] | None:
    """Select the single train/held-out pair, or skip an incomplete pair."""
    current = _heldout(data, prediction, split)
    index = np.flatnonzero((split == "train").to_numpy())
    if current is None or not len(index):
        return None
    part, heldout, current_prediction = current
    return (
        part,
        data.take(index),
        prediction[row_indexer(index, len(data))],
        heldout,
        current_prediction,
    )


def data_comparison_pair(data: Pool, split: pl.Series) -> tuple[str, Pool, Pool] | None:
    """Select train and a single held-out sample without requiring predictions."""
    validate_split_labels(data, split)
    train = np.flatnonzero((split == "train").to_numpy())
    current = np.flatnonzero((split != "train").to_numpy())
    if not len(train) or not len(current):
        return None
    return str(split[int(current[0])]), data.take(train), data.take(current)


def write_heldout_plot(
    data: Pool,
    prediction: FloatArray,
    split: pl.Series,
    root: Path,
    figure: FigureConfig,
    filename: str,
    draw: Callable[..., object],
    **kwargs: object,
) -> list[Path]:
    """Draw one figure from the held-out rows into ``<split>/<filename>``."""
    pair = _heldout(data, prediction, split)
    if pair is None:
        return []
    part, current, predicted = pair
    path = root / part / f"{filename}.{figure.format}"
    draw(
        current.y.to_numpy(),
        predicted,
        save_fig=path,
        weight=current.weight.to_numpy(),
        figure=figure,
        **kwargs,
    )
    return [path]


def validate_report_metrics(values: dict[str, float]) -> dict[str, float]:
    """Return scalar report metrics after checking names and finiteness."""
    metrics: dict[str, float] = {}
    for key, value in values.items():
        if (
            not re.fullmatch(r"[\w .\-/]+", key)
            or not key.strip()
            or any(part in ("", ".", "..") for part in key.split("/"))
        ):
            raise ValueError(f"Invalid report metric name: {key!r}")
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError(f"Report metric {key!r} must be finite")
        metrics[key] = numeric
    return metrics


@dataclass(frozen=True)
class MetricScore:
    """One named metric result; ``value`` is None when the score is not finite."""

    name: str
    value: float | None


def metric_scores(scores: Mapping[str, float]) -> list[MetricScore]:
    """Validate metric names and turn nonfinite scores into missing values."""
    if not scores or any(type(name) is not str or not name.strip() for name in scores):
        raise ValueError("Metric must return nonempty named scores")
    results: list[MetricScore] = []
    for name, score in scores.items():
        value = float(score)
        results.append(MetricScore(name, value if np.isfinite(value) else None))
    return results


def long_metric_rows(
    base: dict[str, object], scores: Mapping[str, float]
) -> list[dict[str, object]]:
    """Expand one window or group into a row per metric with its status."""
    return [
        {
            **base,
            "metric": score.name,
            "value": score.value,
            "status": "ok" if score.value is not None else "undefined",
        }
        for score in metric_scores(scores)
    ]


def expand_skipped_rows(
    records: list[dict[str, object]], names: set[str]
) -> list[dict[str, object]]:
    """Give skipped windows or groups a row for every known metric name."""
    expanded: list[dict[str, object]] = []
    for record in records:
        if record["metric"] is None and names:
            expanded.extend({**record, "metric": name} for name in sorted(names))
        else:
            expanded.append(record)
    return expanded
