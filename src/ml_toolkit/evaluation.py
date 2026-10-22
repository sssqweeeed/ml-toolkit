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
from .config import FigureConfig
from .data import Pool
from .feature_levels import FeatureLevelConfig, LevelTransformer
from .metrics.ranking import ranking_metrics
from .plots import classification, monitoring, pu, ranking, regression
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
    if (
        len(split) != len(data)
        or split.dtype != pl.String
        or split.null_count()
        or not split.is_in(("train", "val", "test")).all()
    ):
        raise ValueError(
            "split must contain aligned non-null String train/val/test labels"
        )
    if {"val", "test"} <= set(split.unique()):
        raise ValueError(
            "Reporting accepts only one held-out split: val or test, not both"
        )


def _heldout(
    data: Pool, prediction: FloatArray, split: pl.Series
) -> tuple[str, Pool, FloatArray] | None:
    validate_inputs(data, prediction, split)
    index = np.flatnonzero((split != "train").to_numpy())
    if not len(index):
        return None
    part = str(split[int(index[0])])
    return part, data.take(index), prediction[index]


def comparison_pair(
    data: Pool, prediction: FloatArray, split: pl.Series
) -> tuple[str, Pool, FloatArray, Pool, FloatArray] | None:
    """Select the single train/held-out pair, or skip an incomplete pair."""
    current = _heldout(data, prediction, split)
    index = np.flatnonzero((split == "train").to_numpy())
    if current is None or not len(index):
        return None
    part, heldout, current_prediction = current
    return part, data.take(index), prediction[index], heldout, current_prediction


def feature_definitions(
    data: Pool, features: tuple[FeatureLevelConfig, ...] | None
) -> tuple[FeatureLevelConfig, ...]:
    if features is not None:
        return features
    return tuple(
        FeatureLevelConfig(
            name,
            (
                "categorical"
                if name in data.cat_features or not dtype.is_numeric()
                else "numerical"
            ),
        )
        for name, dtype in data.X.schema.items()
    )


def feature_filename(index: int, feature: str, format: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", feature).strip("._")[:60] or "feature"
    return f"{index:03d}_{safe}.{format}"


def _validate_features(features: tuple[str, ...] | None) -> None:
    if features is not None and (
        any(not name.strip() for name in features)
        or len(set(features)) != len(features)
    ):
        raise ValueError("features must contain unique nonempty names")


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
        _validate_features(self.features)

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
        transformer = LevelTransformer(definitions, n_bins=n_bins).fit(train.X)
        reference_levels = transformer.transform(train.X)
        current_levels = transformer.transform(current.X)
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
    """Compare weighted time summaries, optionally grouped by business_type.

    Parameters
    ----------
    name : str, default 'time_observed_fitted'
        Unique report name and artifact namespace.
    period : TimePeriod, default 'month'
        Calendar grouping from minute through year, in the data timezone.

    Notes
    -----
    Compares train with validation or test. Pool.business_type supplies a separate
    panel for each group when present. Observed/fitted means and volumes use row weights.
    Timestamps must be ordered across the pair; a shared endpoint is allowed.
    Saves ``time.<format>``. Shared feature rules and n_bins are unused.
    """

    name: str = "time_observed_fitted"
    period: TimePeriod = "month"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.period not in PERIOD_INTERVALS:
            raise ValueError("Unsupported time period")

    def parameters(self) -> dict[str, object]:
        return {"period": self.period}

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
            reference_group=train.business_type,
            current_group=current.business_type,
            period=self.period,
            figure=figure,
        )
        return [path]


@dataclass(frozen=True, kw_only=True)
class NumericalFeatureDistributions(PredictionEvaluator):
    """Compare unweighted numerical time distributions on original features.

    Parameters
    ----------
    name : str, default 'numerical_feature_distributions'
        Unique report name and artifact namespace.
    features : tuple of str or None, default None
        Numeric feature names to display. None uses numeric columns in Pool.X;
        an empty tuple selects none. Missing or nonnumeric names raise ValueError.
    period : TimePeriod, default 'month'
        Calendar grouping in the data timezone.
    quantile_bands : tuple of pairs, default ((0.05, 0.95), (0.25, 0.75))
        Lower and upper quantiles for shaded distribution bands.
    show_mean, show_median : bool, default True
        Whether to draw each central summary.

    Notes
    -----
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

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_features(self.features)
        monitoring.validate_distribution(self.quantile_bands)
        if self.period not in PERIOD_INTERVALS:
            raise ValueError("Unsupported time period")

    def parameters(self) -> dict[str, object]:
        return {
            "features": self.features,
            "period": self.period,
            "quantile_bands": self.quantile_bands,
            "show_mean": self.show_mean,
            "show_median": self.show_median,
        }

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
        _, train, _, current, _ = pair
        names = self.features
        if names is None:
            names = tuple(
                name for name, dtype in train.X.schema.items() if dtype.is_numeric()
            )
        time_column = "__ml_toolkit_plot_time__"
        while time_column in train.X.columns:
            time_column += "_"
        reference = train.X.with_columns(train.time.alias(time_column))
        heldout = current.X.with_columns(current.time.alias(time_column))
        for index, name in enumerate(names):
            if name not in train.X.columns or not train.X.schema[name].is_numeric():
                raise ValueError(f"Feature {name!r} is missing or not numerical")
            path = artifact_dir / feature_filename(index, name, figure.format)
            monitoring.time_feature_distribution(
                reference,
                heldout,
                name,
                time_column,
                save_fig=path,
                period=self.period,
                quantile_bands=self.quantile_bands,
                show_mean=self.show_mean,
                show_median=self.show_median,
                figure=figure,
            )
            paths.append(path)
        return paths


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
    "NumericalFeatureDistributions",
]
