"""Observed-versus-fitted reports by feature level and calendar period."""

from dataclasses import dataclass
from pathlib import Path
import polars as pl
from .base import PredictionEvaluator
from .._time import PERIOD_INTERVALS
from .._grouping import row_groups, validate_grouping
from ..config import FigureConfig
from ..data import Pool
from ..feature_levels import FeatureLevelConfig, LevelTransformer
from ..feature_levels.config import feature_definitions, validate_features
from ..plots import regression
from ..plots.utils import feature_filename
from ..types import FloatArray, TimePeriod

from ._common import comparison_pair


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
