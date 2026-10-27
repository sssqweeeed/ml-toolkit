"""Composable prediction reports, independent of training-stage scheduling.

Each call receives aligned rows, predictions and a String series containing
train and at most one held-out split (val or test). Inputs must not be mutated.
Artifact paths are relative to the supplied temporary directory (absolute
paths inside it are also valid).
Instances are reused across fits; bundled reports retain settings only.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import polars as pl

from ._time import PERIOD_INTERVALS
from ._rows import row_indexer
from ._grouping import row_groups, validate_grouping
from .config import FigureConfig
from .data import Pool
from .data_analysis import validate_split
from .data_analysis import (
    NumericalFeatureDistributions as _NumericalFeatureDistributions,
)
from .feature_levels import FeatureLevelConfig, LevelTransformer
from .feature_levels.config import feature_definitions, validate_features
from .metrics.ranking import ranking_metrics
from .metrics import Metric
from .plots import classification, pu, ranking, regression
from .plots.utils import feature_filename
from .types import FloatArray, TimePeriod


@dataclass(frozen=True, kw_only=True)
class PredictionEvaluator:
    """Named additional metrics and artifacts; neither method is mandatory.

    ``evaluate`` keys are logged under ``<name>/<key>`` and do not become CV
    averages or optimization objectives. Return finite scalar values only.
    ``write_artifacts`` receives shared figure and feature-level settings.
    Only returned files are uploaded under ``<name>/``. The directory is deleted
    after uploading.
    """

    name: str

    def __post_init__(self) -> None:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.name) is None:
            raise ValueError(
                "Evaluator name must start with an ASCII letter or digit and contain only ASCII letters, digits, underscores, hyphens or dots"
            )

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
            when relevant; these values do not become optimization objectives.
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


def validate_inputs(data: Pool, prediction: FloatArray, split: pl.Series) -> None:
    if prediction.shape != (len(data),) or not np.isfinite(prediction).all():
        raise ValueError("Predictions must contain one finite value per row")
    validate_split(data, split)


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


def _single(
    data: Pool,
    prediction: FloatArray,
    split: pl.Series,
    root: Path,
    figure: FigureConfig,
    filename: str,
    draw: Callable[..., object],
    **kwargs: object,
) -> list[Path]:
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


def _gains(
    observed: FloatArray,
    prediction: FloatArray,
    *,
    weight: FloatArray,
    save_fig: Path,
    figure: FigureConfig,
) -> None:
    if not np.all(observed >= 0) or np.sum(observed * weight) <= 0:
        raise ValueError(
            "Gains/lift require nonnegative targets with positive weighted total"
        )
    ranking.gains_curve(
        ranking_metrics(observed, prediction, weight), save_fig=save_fig, figure=figure
    )


def _lift(
    observed: FloatArray,
    prediction: FloatArray,
    *,
    weight: FloatArray,
    save_fig: Path,
    figure: FigureConfig,
) -> None:
    if not np.all(observed >= 0) or np.sum(observed * weight) <= 0:
        raise ValueError(
            "Gains/lift require nonnegative targets with positive weighted total"
        )
    ranking.lift_curve(
        ranking_metrics(observed, prediction, weight), save_fig=save_fig, figure=figure
    )


def _pu_distribution(
    observed: FloatArray,
    prediction: FloatArray,
    *,
    weight: FloatArray,
    save_fig: Path,
    figure: FigureConfig,
    histogram_bins: int,
) -> None:
    pu.score_distribution(
        observed,
        prediction,
        weights=weight,
        save_fig=save_fig,
        figure=figure,
        histogram_bins=histogram_bins,
    )


def _pu_capture(
    observed: FloatArray,
    prediction: FloatArray,
    *,
    weight: FloatArray,
    save_fig: Path,
    figure: FigureConfig,
) -> None:
    pu.positive_capture(
        observed, prediction, weights=weight, save_fig=save_fig, figure=figure
    )


@dataclass(frozen=True, kw_only=True)
class PrecisionRecall(PredictionEvaluator):
    """Plot weighted precision against recall on held-out probabilities.

    Parameters
    ----------
    name : str, default 'precision_recall'
        Unique report name and artifact namespace.
    max_curve_points : int, default 1000
        Maximum displayed threshold points; does not change metric calculations.

    Notes
    -----
    Uses the single validation or test sample, ignoring train rows. Targets are
    confirmed binary labels and predictions are probabilities in [0, 1].
    Saves ``<split>/precision_recall.<format>``. Row weights affect both curves.
    """

    name: str = "precision_recall"
    max_curve_points: int = 1000

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.max_curve_points < 1:
            raise ValueError("max_curve_points must be positive")

    def parameters(self) -> dict[str, object]:
        return {"max_curve_points": self.max_curve_points}

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
        return _single(
            data,
            prediction,
            split,
            artifact_dir,
            figure,
            "precision_recall",
            classification.precision_recall,
            max_curve_points=self.max_curve_points,
        )


@dataclass(frozen=True, kw_only=True)
class F1Threshold(PredictionEvaluator):
    """Plot weighted F1 against held-out probability thresholds.

    Parameters
    ----------
    name : str, default 'f1_threshold'
        Unique report name and artifact namespace.
    max_curve_points : int, default 1000
        Maximum displayed threshold points.

    Notes
    -----
    Uses confirmed binary validation or test labels and probabilities in [0, 1].
    Saves ``<split>/f1_threshold.<format>`` without changing the model threshold.
    """

    name: str = "f1_threshold"
    max_curve_points: int = 1000

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.max_curve_points < 1:
            raise ValueError("max_curve_points must be positive")

    def parameters(self) -> dict[str, object]:
        return {"max_curve_points": self.max_curve_points}

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
        return _single(
            data,
            prediction,
            split,
            artifact_dir,
            figure,
            "f1_threshold",
            classification.f1_threshold,
            max_curve_points=self.max_curve_points,
        )


@dataclass(frozen=True, kw_only=True)
class Calibration(PredictionEvaluator):
    """Compare held-out probabilities with observed positive rates.

    Parameters
    ----------
    name : str, default 'calibration'
        Unique report name and artifact namespace.
    n_bins : int, default 10
        Number of probability quantiles for the calibration curve. Independent
        of shared feature-level binning.
    histogram_bins : int, default 30
        Number of bins in the accompanying probability histogram.

    Notes
    -----
    Uses confirmed binary labels and probabilities in [0, 1]. Means and the
    histogram use weights; quantile boundaries use positive-weight rows without
    weighting. Saves ``<split>/calibration.<format>`` for validation or test.
    """

    name: str = "calibration"
    n_bins: int = 10
    histogram_bins: int = 30

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_bins < 1:
            raise ValueError("n_bins must be positive")
        if self.histogram_bins < 1:
            raise ValueError("histogram_bins must be positive")

    def parameters(self) -> dict[str, object]:
        return {"n_bins": self.n_bins, "histogram_bins": self.histogram_bins}

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
        return _single(
            data,
            prediction,
            split,
            artifact_dir,
            figure,
            "calibration",
            classification.calibration,
            n_bins=self.n_bins,
            histogram_bins=self.histogram_bins,
        )


@dataclass(frozen=True, kw_only=True)
class Gains(PredictionEvaluator):
    """Plot captured observed outcome against held-out exposure capacity.

    Parameters
    ----------
    name : str, default 'gains'
        Unique report name and artifact namespace.

    Notes
    -----
    Ranks validation or test rows by descending prediction. Pool weights supply
    exposure; targets must be nonnegative with positive weighted total. The
    curve shows cumulative ``target * weight`` share, including the default 10%
    capacity point. Saves ``<split>/gains.<format>``.
    """

    name: str = "gains"

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
        return _single(data, prediction, split, artifact_dir, figure, "gains", _gains)


@dataclass(frozen=True, kw_only=True)
class Lift(PredictionEvaluator):
    """Plot held-out outcome capture relative to uniform selection.

    Parameters
    ----------
    name : str, default 'lift'
        Unique report name and artifact namespace.

    Notes
    -----
    Lift divides captured weighted outcome share by exposure share after
    ranking by prediction. Uses validation or test rows and the default 10%
    capacity point. Targets must be nonnegative with positive weighted total.
    Saves ``<split>/lift.<format>``.
    """

    name: str = "lift"

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
        return _single(data, prediction, split, artifact_dir, figure, "lift", _lift)


@dataclass(frozen=True, kw_only=True)
class PUScoreDistribution(PredictionEvaluator):
    """Compare score distributions for known positives and unlabeled rows.

    Parameters
    ----------
    name : str, default 'pu_score_distribution'
        Unique report name and artifact namespace.
    histogram_bins : int, default 30
        Number of score bins.

    Notes
    -----
    Labels 1 and 0 mean known positive and unlabeled. Each group's weighted
    histogram is normalized separately; both need positive total weight.
    Uses validation or test rows and saves
    ``<split>/pu_score_distribution.<format>``.
    """

    name: str = "pu_score_distribution"
    histogram_bins: int = 30

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.histogram_bins < 1:
            raise ValueError("histogram_bins must be positive")

    def parameters(self) -> dict[str, object]:
        return {"histogram_bins": self.histogram_bins}

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
        return _single(
            data,
            prediction,
            split,
            artifact_dir,
            figure,
            "pu_score_distribution",
            _pu_distribution,
            histogram_bins=self.histogram_bins,
        )


@dataclass(frozen=True, kw_only=True)
class PUPositiveCapture(PredictionEvaluator):
    """Plot known-positive capture against unlabeled review capacity.

    Parameters
    ----------
    name : str, default 'pu_positive_capture'
        Unique report name and artifact namespace.

    Notes
    -----
    Labels 1 and 0 mean known positive and unlabeled. Both axes use weighted
    percentages; tied scores enter together. Capture is not recall for all
    positives without assumptions about labeling. Uses validation or test rows
    and saves ``<split>/pu_positive_capture.<format>``.
    """

    name: str = "pu_positive_capture"

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
        return _single(
            data,
            prediction,
            split,
            artifact_dir,
            figure,
            "pu_positive_capture",
            _pu_capture,
        )


@dataclass(frozen=True, kw_only=True)
class ObservedFitted(PredictionEvaluator):
    """Compare weighted observed/fitted means on train-fitted feature levels.

    Parameters
    ----------
    name : str, default 'observed_fitted'
        Unique report name and artifact namespace.
    features : tuple of str or None, default None
        Names to display from shared feature definitions. None selects all;
        an empty tuple selects none. Unknown names raise ValueError.

    Notes
    -----
    Fits LevelTransformer on train using the supplied feature rules and n_bins.
    Each figure compares train with validation or test using the same levels.
    Lines show weighted means; volume is summed weight. Zero-weight or absent
    levels have no mean. Saves one indexed feature filename in the report root.
    """

    name: str = "observed_fitted"
    features: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        validate_features(self.features)

    def parameters(self) -> dict[str, object]:
        return {"features": self.features}

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
        paths: list[Path] = []
        pair = comparison_pair(data, prediction, split)
        if pair is None:
            return []
        part, train, train_pred, current, current_pred = pair
        definitions = feature_definitions(train, features)
        if not definitions:
            return []
        if self.features is not None:
            if not set(self.features) <= {item.name for item in definitions}:
                raise ValueError(
                    "ObservedFitted features must exist in shared feature definitions"
                )
            by_name = {item.name: item for item in definitions}
            definitions = tuple(by_name[name] for name in self.features)
        if not definitions:
            return []
        transformer = LevelTransformer(definitions, n_bins=n_bins).fit(
            train.to_analysis_frame()
        )
        reference_levels = transformer.transform(train.to_analysis_frame())
        current_levels = transformer.transform(current.to_analysis_frame())
        assert transformer.feature_levels is not None
        for index, level in enumerate(transformer.feature_levels):
            path = artifact_dir / feature_filename(
                index, level.config.name, figure.format
            )
            regression.observed_fitted_by_level(
                reference_levels[level.config.name],
                train.y.to_numpy(),
                train_pred,
                current_levels[level.config.name],
                current.y.to_numpy(),
                current_pred,
                save_fig=path,
                level_order=level.levels,
                reference_weight=train.weight.to_numpy(),
                current_weight=current.weight.to_numpy(),
                feature_name=level.config.name,
                current_label="Test" if part == "test" else "Validation",
                figure=figure,
            )
            paths.append(path)
        return paths


@dataclass(frozen=True, kw_only=True)
class TimeObservedFitted(PredictionEvaluator):
    """Compare weighted time summaries, optionally grouped by exact values.

    Parameters
    ----------
    name : str, default 'time_observed_fitted'
        Unique report name and artifact namespace.
    period : TimePeriod, default 'month'
        Calendar grouping from minute through year, in the data timezone.
    by : tuple of str, default ()
        Model or auxiliary columns defining exact-value groups. Multiple columns
        select combinations. Empty gives an overall plot. Precompute bands for
        continuous values; no numerical levels are fitted here.
    max_groups : int, default 50
        Maximum distinct groups across both samples, including missing groups.

    Notes
    -----
    Compares train with validation or test, with a panel for each requested group.
    Observed/fitted means and volumes use row weights.
    Timestamps must be ordered across the pair; a shared endpoint is allowed.
    Saves ``time.<format>``. Shared feature rules and n_bins are unused.
    """

    name: str = "time_observed_fitted"
    period: TimePeriod = "month"
    by: tuple[str, ...] = ()
    max_groups: int = 50

    def __post_init__(self) -> None:
        super().__post_init__()
        validate_grouping(self.by, self.max_groups, required=False)
        if self.period not in PERIOD_INTERVALS:
            raise ValueError("Unsupported time period")

    def parameters(self) -> dict[str, object]:
        return {"period": self.period, "by": self.by, "max_groups": self.max_groups}

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
        pair = comparison_pair(data, prediction, split)
        if pair is None:
            return []
        _, train, train_pred, current, current_pred = pair
        reference_group = current_group = None
        if self.by:
            labels = [""] * len(data)
            for group in row_groups(data, self.by, self.max_groups):
                for index in group.indices:
                    labels[int(index)] = group.label
            groups = pl.Series(labels)
            reference_group = groups.filter(split == "train")
            current_group = groups.filter(split != "train")
        path = artifact_dir / f"time.{figure.format}"
        regression.time_observed_fitted(
            train.time.to_list(),
            train.y.to_numpy(),
            train_pred,
            current.time.to_list(),
            current.y.to_numpy(),
            current_pred,
            save_fig=path,
            reference_weight=train.weight.to_numpy(),
            current_weight=current.weight.to_numpy(),
            reference_group=reference_group,
            current_group=current_group,
            period=self.period,
            figure=figure,
        )
        return [path]


@dataclass(frozen=True, kw_only=True)
class GroupedMetrics(PredictionEvaluator):
    """Calculate an ordinary metric separately for exact-value groups and splits.

    Parameters
    ----------
    metric : Metric
        Callable accepting aligned target, prediction and weight arrays and
        returning an object with to_dict(). Exceptions propagate unchanged.
    by : tuple of str
        Nonempty model/auxiliary grouping columns. Multiple names select
        combinations. Prepare a categorical band column for continuous values.
    name : str, default 'grouped_metrics'
        Unique report name and artifact namespace.
    min_rows : int, default 1
        Minimum number of positive-weight rows required to calculate metrics.
    max_groups : int, default 50
        Maximum distinct groups across all supplied splits, including missing.

    Notes
    -----
    calculate() returns a typed long table, including empty split/group pairs.
    Statuses are ok, empty, zero_weight, insufficient_rows, or undefined.
    Missing results have null values. Only finite results are logged by evaluate.
    write_artifacts saves metrics.parquet, preserving typed group values and IDs.
    Instances retain settings only and can be reused across folds.
    """

    metric: Metric
    by: tuple[str, ...]
    name: str = "grouped_metrics"
    min_rows: int = 1
    max_groups: int = 50

    def __post_init__(self) -> None:
        super().__post_init__()
        validate_grouping(self.by, self.max_groups, required=True)
        if type(self.min_rows) is not int or self.min_rows < 1:
            raise ValueError("min_rows must be a positive integer")
        reserved = {
            "split",
            "group_id",
            "n_rows",
            "n_positive_weight_rows",
            "sum_weight",
            "metric",
            "value",
            "status",
        }
        if reserved.intersection(self.by):
            raise ValueError("Grouping column names conflict with metric table columns")

    def parameters(self) -> dict[str, object]:
        metric_type = type(self.metric)
        identifier = (
            f"{getattr(self.metric, '__module__', metric_type.__module__)}."
            f"{getattr(self.metric, '__qualname__', metric_type.__qualname__)}"
        )
        return {
            "metric": identifier,
            "by": self.by,
            "min_rows": self.min_rows,
            "max_groups": self.max_groups,
        }

    def calculate(
        self, *, data: Pool, prediction: FloatArray, split: pl.Series
    ) -> pl.DataFrame:
        """Return per-split/group scores with typed keys, support counts and status.

        Groups are determined across the supplied rows. Each represented split
        receives every group, including those with no rows in that split.
        Skipped groups have metric=None when no metric names can be determined.
        Nonfinite scores become null/undefined; metric exceptions are not hidden.
        """
        validate_inputs(data, prediction, split)
        groups = row_groups(data, self.by, self.max_groups)
        target = data.y.to_numpy().astype(float)
        weight = data.weight.to_numpy().astype(float)
        if not np.isfinite(target).all():
            raise ValueError("Target values must be finite")
        if not np.isfinite(weight).all() or (weight < 0).any():
            raise ValueError("Weights must be finite and nonnegative")
        records: list[dict[str, object]] = []
        metric_names: set[str] = set()
        for part in ("train", "val", "test"):
            selected = (split == part).to_numpy()
            if not selected.any():
                continue
            for group in groups:
                index = group.indices[selected[group.indices]]
                positive = int(np.count_nonzero(weight[index] > 0))
                total = float(weight[index].sum())
                if not len(index):
                    status = "empty"
                elif positive == 0:
                    status = "zero_weight"
                elif positive < self.min_rows:
                    status = "insufficient_rows"
                else:
                    status = "ok"
                base: dict[str, object] = {
                    **group.key.row(0, named=True),
                    "group_id": group.identifier,
                    "split": part,
                    "n_rows": len(index),
                    "n_positive_weight_rows": positive,
                    "sum_weight": total,
                }
                if status != "ok":
                    records.append(
                        {**base, "metric": None, "value": None, "status": status}
                    )
                    continue
                scores = self.metric(
                    target[index], prediction[index], weight[index]
                ).to_dict()
                if not scores:
                    raise ValueError(
                        "Grouped metric must return at least one named score"
                    )
                for name, score in scores.items():
                    if type(name) is not str or not name.strip():
                        raise ValueError(
                            "Grouped metric names must be nonempty strings"
                        )
                    metric_names.add(name)
                    value = float(score)
                    finite = np.isfinite(value)
                    records.append(
                        {
                            **base,
                            "metric": name,
                            "value": value if finite else None,
                            "status": "ok" if finite else "undefined",
                        }
                    )
        expanded: list[dict[str, object]] = []
        for record in records:
            if record["metric"] is None and metric_names:
                expanded.extend(
                    {**record, "metric": name} for name in sorted(metric_names)
                )
            else:
                expanded.append(record)
        schema = dict(data.to_analysis_frame(self.by).schema)
        schema.update(
            {
                "group_id": pl.String(),
                "split": pl.String(),
                "n_rows": pl.Int64(),
                "n_positive_weight_rows": pl.Int64(),
                "sum_weight": pl.Float64(),
                "metric": pl.String(),
                "value": pl.Float64(),
                "status": pl.String(),
            }
        )
        return pl.DataFrame(expanded, schema=schema)

    def evaluate(
        self, *, data: Pool, prediction: FloatArray, split: pl.Series
    ) -> dict[str, float]:
        table = self.calculate(data=data, prediction=prediction, split=split)
        scores: dict[str, float] = {}
        for row in table.filter(pl.col("status") == "ok").iter_rows(named=True):
            name = str(row["metric"])
            # Ordinary names stay readable; hex encoding is injective and uses
            # a reserved leading underscore, so unsafe names cannot collide.
            key = (
                name
                if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)
                else "_hex_" + name.encode().hex()
            )
            scores[f"{row['split']}/{row['group_id']}/{key}"] = float(row["value"])
        return scores

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
        table = self.calculate(data=data, prediction=prediction, split=split)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / "metrics.parquet"
        table.write_parquet(path)
        return [path]


@dataclass(frozen=True, kw_only=True)
class NumericalFeatureDistributions(PredictionEvaluator):
    """Compare unweighted numerical time distributions on original features.

    Parameters
    ----------
    name : str, default 'numerical_feature_distributions'
        Unique report name and artifact namespace.
    features : tuple of str or None, default None
        Numeric feature names to display. None uses model and auxiliary numeric columns;
        an empty tuple selects none. Missing or nonnumeric names raise ValueError.
    period : TimePeriod, default 'month'
        Calendar grouping in the data timezone.
    quantile_bands : tuple of pairs, default ((0.05, 0.95), (0.25, 0.75))
        Lower and upper quantiles for shaded distribution bands.
    show_mean, show_median : bool, default True
        Whether to draw each central summary.

    Notes
    -----
    Compatibility adapter for existing prediction_evaluators calls. New code
    should import this report from ml_toolkit.data_analysis and pass it through
    data_analyzers instead.

    Compares train with validation or test using unweighted original values,
    independent of predictions, Pool.weight, shared feature rules, and n_bins.
    Nulls and non-finite values are omitted. Saves one indexed feature
    filename in the report root.
    """

    name: str = "numerical_feature_distributions"
    features: tuple[str, ...] | None = None
    period: TimePeriod = "month"
    quantile_bands: tuple[tuple[float, float], ...] = ((0.05, 0.95), (0.25, 0.75))
    show_mean: bool = True
    show_median: bool = True

    def _analyzer(self) -> _NumericalFeatureDistributions:
        return _NumericalFeatureDistributions(
            name=self.name,
            features=self.features,
            period=self.period,
            quantile_bands=self.quantile_bands,
            show_mean=self.show_mean,
            show_median=self.show_median,
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        self._analyzer()

    def parameters(self) -> dict[str, object]:
        return self._analyzer().parameters()

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
        validate_inputs(data, prediction, split)
        return self._analyzer().write_artifacts(
            data=data,
            split=split,
            artifact_dir=artifact_dir,
            figure=figure,
            features=features,
            n_bins=n_bins,
        )


__all__ = [
    "PredictionEvaluator",
    "PrecisionRecall",
    "F1Threshold",
    "Calibration",
    "Gains",
    "Lift",
    "PUScoreDistribution",
    "PUPositiveCapture",
    "ObservedFitted",
    "TimeObservedFitted",
    "GroupedMetrics",
    "NumericalFeatureDistributions",
]
