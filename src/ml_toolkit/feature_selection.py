"""Feature selectors and their complete, model-free search histories."""

from __future__ import annotations

import logging

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from math import isfinite
from time import perf_counter
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias

import numpy as np

from .data import validate_training_data, validate_training_categories
from .metrics import Metric
from .models.base import Model
from .types import ModelParams

if TYPE_CHECKING:
    from .training.contracts import FeatureRanker, Pool

logger = logging.getLogger(__name__)

SelectionSettings: TypeAlias = dict[
    str, str | int | float | bool | None | tuple[str, ...]
]
"""Flat, JSON-compatible selector configuration, without executable objects."""

__all__ = [
    "FeatureSelector",
    "RFESelector",
    "TopKSelector",
    "ForwardSelector",
    "FeatureSelectionResult",
    "FeatureSelectionStep",
    "SelectionSettings",
]


@dataclass(frozen=True)
class FeatureSelectionStep:
    """Store one fitted and evaluated feature subset.

    Attributes
    ----------
    features : tuple[str, ...]
        Evaluated input columns in original order.
    metrics : dict[str, float]
        All supplied validation scores returned by the training metric.
    importances : dict[str, float]
        Ranking values by feature name; empty when no ranking is performed.
    eliminated_features : tuple[str, ...]
        Columns removed by an elimination step; empty for forward candidates.
    fit_seconds : float
        Elapsed adapter construction and training time.
    evaluation_seconds : float
        Elapsed prediction and metric calculation time.
    ranking_seconds : float
        Elapsed ranking and importance validation time, or zero when no ranking is performed.
    best_iteration : int or None
        Adapter's zero-based best iteration; not used for final fitting.
    """

    features: tuple[str, ...]
    metrics: dict[str, float]
    importances: dict[str, float]
    eliminated_features: tuple[str, ...]
    fit_seconds: float
    evaluation_seconds: float
    ranking_seconds: float
    best_iteration: int | None


@dataclass(frozen=True)
class FeatureSelectionResult:
    """Store a selector's selected subset and complete search history.

    Attributes
    ----------
    features : tuple[str, ...]
        Selected columns in original input order; may include the full set.
    selected_step : int
        Zero-based index into steps for the selected subset.
    steps : tuple[FeatureSelectionStep, ...]
        Chronological history of every fitted subset; contains no fitted models.

    settings : SelectionSettings
        Flat configuration with string keys and scalar or string-tuple values;
        excludes models and rankers.

    Notes
    -----
    Validation scores describe selection, not independent performance.
    Frozen result objects contain mutable metric and importance dictionaries.
    """

    features: tuple[str, ...]
    selected_step: int
    steps: tuple[FeatureSelectionStep, ...]
    settings: SelectionSettings = field(default_factory=lambda: {})


class FeatureSelector(Protocol):
    """Select features independently on each supplied training/validation split.

    Notes
    -----
    Implement ``select`` without retaining learned state between calls. Use a
    fresh model_factory(params.copy()) for every fit. Return every evaluated
    subset and concrete settings, never fitted models or executable objects.
    The workflow excludes test rows and reuses the final split's selection.
    """

    def select(
        self,
        *,
        train: Pool,
        validation: Pool,
        params: ModelParams,
        model_factory: Callable[[ModelParams], Model],
        metric: Metric,
    ) -> FeatureSelectionResult:
        """Return a selected subset and its complete search history.

        Parameters
        ----------
        train, validation : Pool
            Aligned prepared data with identical feature names and order.
        params : ModelParams
            Base constructor parameters; copy before passing to a fresh model.
        model_factory : callable
            Constructs an independent model for every fit.
        metric : Metric
            Evaluates targets, predictions and weights on validation rows.

        Returns
        -------
        FeatureSelectionResult
            Nonempty unique input features in original order, their selected
            history index, all evaluated subsets and serializable settings.
            Scores guide selection and are not independent quality estimates.
        """
        ...


def _validate_options(limit: int, protected: tuple[str, ...]) -> None:
    if limit < 1:
        raise ValueError("Feature count and fit budgets must be positive")
    if len(protected) > limit:
        raise ValueError("protected_features must not exceed the feature limit")
    if len(set(protected)) != len(protected) or any(
        not name.strip() for name in protected
    ):
        raise ValueError("protected_features must contain unique nonempty names")


def _validate_objective(metric: str, direction: str, tolerance: float) -> None:
    if not metric.strip():
        raise ValueError("metric must not be empty")
    if direction not in ("minimize", "maximize"):
        raise ValueError("direction must be minimize or maximize")
    if not isfinite(tolerance) or tolerance < 0:
        raise ValueError("Score tolerance/improvement must be finite and nonnegative")


def _validate_data(
    train: Pool, validation: Pool, limit: int, protected: tuple[str, ...]
) -> None:
    validate_training_data(train)
    validate_training_data(validation)
    validate_training_categories(train)
    validate_training_categories(validation)
    if train.X.columns != validation.X.columns:
        raise ValueError("Train and validation features must match")
    if limit > len(train.X.columns):
        raise ValueError("Feature limit must not exceed the input feature count")
    missing = set(protected) - set(train.X.columns)
    if missing:
        raise ValueError(f"Protected features are missing from X: {sorted(missing)}")


def _evaluate(
    names: tuple[str, ...],
    train: Pool,
    validation: Pool,
    params: ModelParams,
    model_factory: Callable[[ModelParams], Model],
    metric: Metric,
    objective: str | None = None,
    ranker: FeatureRanker | None = None,
) -> FeatureSelectionStep:
    train = train.select_features(names)
    validation = validation.select_features(names)
    started = perf_counter()
    model = model_factory(params.copy()).fit(train, validation)
    fit_seconds = perf_counter() - started
    started = perf_counter()
    prediction = model.predict_pool(validation)
    if prediction.shape != (len(validation),) or not np.isfinite(prediction).all():
        raise ValueError("Predictions must contain one finite value per row")
    scores = dict(
        metric(
            validation.y.to_numpy().astype(float),
            prediction,
            validation.weight.to_numpy().astype(float),
        ).to_dict()
    )
    if objective is not None:
        if objective not in scores:
            raise ValueError(
                f"Metric {objective!r} is unavailable. Available metrics: {sorted(scores)}"
            )
        if not np.isfinite(scores[objective]):
            raise ValueError(f"Selection metric {objective!r} must be finite")
    evaluation_seconds = perf_counter() - started
    importances: dict[str, float] = {}
    ranking_seconds = 0.0
    if ranker is not None:
        started = perf_counter()
        importances = dict(ranker(model, validation))
        if set(importances) != set(names):
            raise ValueError(
                "Feature ranker must return exactly one importance per input column"
            )
        if not all(np.isfinite(value) for value in importances.values()):
            raise ValueError("Feature importance must be finite")
        importances = {name: float(importances[name]) for name in names}
        ranking_seconds = perf_counter() - started
    return FeatureSelectionStep(
        names,
        scores,
        importances,
        (),
        fit_seconds,
        evaluation_seconds,
        ranking_seconds,
        model.best_iteration,
    )


@dataclass(frozen=True, kw_only=True)
class RFESelector:
    """Eliminate low-ranked features on a geometric schedule.

    Parameters
    ----------
    min_features : int
        Requested minimum subset size, reached when the fit budget permits.
        The selected result can be larger.
    metric : str
        Name of the finite validation score to optimize.
    direction : Literal['minimize', 'maximize']
        Objective direction.
    ranker : FeatureRanker
        Ranks the fitted model on validation rows; higher values are preferred.
    max_fits : int, default 6
        Maximum number of evaluated subsets, including the complete set.
    score_tolerance : float, default 0
        Choose the smallest visited subset within this absolute distance of best.
    protected_features : tuple[str, ...], default ()
        Unique columns retained throughout, counted within min_features.

    Notes
    -----
    Fits fresh models. Equal ranks retain earlier input columns. Does not log
    artifacts itself. Validation scores are search results, not held-out quality.
    """

    min_features: int
    metric: str
    direction: Literal["minimize", "maximize"]
    ranker: FeatureRanker
    max_fits: int = 6
    score_tolerance: float = 0.0
    protected_features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_options(self.min_features, self.protected_features)
        _validate_options(self.max_fits, ())
        _validate_objective(self.metric, self.direction, self.score_tolerance)

    def select(
        self,
        *,
        train: Pool,
        validation: Pool,
        params: ModelParams,
        model_factory: Callable[[ModelParams], Model],
        metric: Metric,
    ) -> FeatureSelectionResult:
        """Return a selected subset and its complete search history.

        Parameters
        ----------
        train, validation : Pool
            Aligned prepared data with identical feature names and order.
        params : ModelParams
            Base constructor parameters; copy before passing to a fresh model.
        model_factory : callable
            Constructs an independent model for every fit.
        metric : Metric
            Evaluates targets, predictions and weights on validation rows.

        Returns
        -------
        FeatureSelectionResult
            Nonempty unique input features in original order, their selected
            history index, all evaluated subsets and serializable settings.
            Scores guide selection and are not independent quality estimates.
        """
        _validate_data(train, validation, self.min_features, self.protected_features)
        names = tuple(train.X.columns)
        count = min(self.max_fits, len(names) - self.min_features + 1)
        sizes = tuple(
            dict.fromkeys(
                int(n)
                for n in np.rint(
                    np.geomspace(len(names), self.min_features, count)
                ).astype(int)
            )
        )
        protected = set(self.protected_features)
        history: list[FeatureSelectionStep] = []
        for index, _ in enumerate(sizes):
            step = _evaluate(
                names,
                train,
                validation,
                params,
                model_factory,
                metric,
                self.metric,
                self.ranker if index + 1 < len(sizes) else None,
            )
            if index + 1 < len(sizes):
                ranked = sorted(
                    (n for n in names if n not in protected),
                    key=step.importances.__getitem__,
                    reverse=True,
                )
                keep = protected | set(ranked[: sizes[index + 1] - len(protected)])
                step = replace(
                    step, eliminated_features=tuple(n for n in names if n not in keep)
                )
                names = tuple(n for n in names if n in keep)
            history.append(step)
        scores = [step.metrics[self.metric] for step in history]
        best = min(scores) if self.direction == "minimize" else max(scores)
        eligible = [
            i
            for i, score in enumerate(scores)
            if (
                score <= best + self.score_tolerance
                if self.direction == "minimize"
                else score >= best - self.score_tolerance
            )
        ]
        selected = min(eligible, key=lambda i: len(history[i].features))
        logger.info(
            "Completed feature selection: %s features selected at step %s",
            len(history[selected].features),
            selected + 1,
        )
        logger.debug("Selected features: %s", history[selected].features)
        return FeatureSelectionResult(
            history[selected].features,
            selected,
            tuple(history),
            {
                "selector": "rfe",
                "min_features": self.min_features,
                "metric": self.metric,
                "direction": self.direction,
                "max_fits": self.max_fits,
                "score_tolerance": self.score_tolerance,
                "protected_features": self.protected_features,
            },
        )


@dataclass(frozen=True, kw_only=True)
class TopKSelector:
    """Retain exactly k features from one ranking of the full fitted model.

    Parameters
    ----------
    k : int
        Selected feature count, including protected features.
    ranker : FeatureRanker
        Ranks the full fitted model on validation rows; higher is better.
    protected_features : tuple[str, ...], default ()
        Unique mandatory columns counted within k.

    Notes
    -----
    Fits the full set and, if different, the selected subset. Keeps k even if
    validation quality worsens. Equal ranks retain earlier input columns.
    """

    k: int
    ranker: FeatureRanker
    protected_features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_options(self.k, self.protected_features)

    def select(
        self,
        *,
        train: Pool,
        validation: Pool,
        params: ModelParams,
        model_factory: Callable[[ModelParams], Model],
        metric: Metric,
    ) -> FeatureSelectionResult:
        """Return a selected subset and its complete search history.

        Parameters
        ----------
        train, validation : Pool
            Aligned prepared data with identical feature names and order.
        params : ModelParams
            Base constructor parameters; copy before passing to a fresh model.
        model_factory : callable
            Constructs an independent model for every fit.
        metric : Metric
            Evaluates targets, predictions and weights on validation rows.

        Returns
        -------
        FeatureSelectionResult
            Nonempty unique input features in original order, their selected
            history index, all evaluated subsets and serializable settings.
            Scores guide selection and are not independent quality estimates.
        """
        _validate_data(train, validation, self.k, self.protected_features)
        names = tuple(train.X.columns)
        full = _evaluate(
            names, train, validation, params, model_factory, metric, ranker=self.ranker
        )
        protected = set(self.protected_features)
        ranked = sorted(
            (n for n in names if n not in protected),
            key=full.importances.__getitem__,
            reverse=True,
        )
        keep = protected | set(ranked[: self.k - len(protected)])
        selected = tuple(n for n in names if n in keep)
        history = [
            replace(full, eliminated_features=tuple(n for n in names if n not in keep))
        ]
        if selected != names:
            history.append(
                _evaluate(selected, train, validation, params, model_factory, metric)
            )
        return FeatureSelectionResult(
            selected,
            len(history) - 1,
            tuple(history),
            {
                "selector": "top_k",
                "k": self.k,
                "protected_features": self.protected_features,
            },
        )


@dataclass(frozen=True, kw_only=True)
class ForwardSelector:
    """Add the best feature while validation improvement exceeds a threshold.

    Parameters
    ----------
    metric : str
        Name of the finite validation objective.
    direction : Literal['minimize', 'maximize']
        Objective direction.
    max_features : int
        Maximum result size including protected features.
    max_fits : int
        Total fit budget, including the protected-only baseline. A candidate
        round starts only if all its remaining features can be tested.
    min_improvement : float, default 0
        Strict minimum absolute improvement required to accept an addition.
    protected_features : tuple[str, ...], default ()
        Mandatory starting subset. Without it, compare every singleton first.

    Notes
    -----
    Insufficient budget for the initial comparison raises ValueError. History
    includes rejected candidates. Equal scores prefer earlier input columns.
    Each call owns all fitted state; only scalar metadata is returned.
    """

    metric: str
    direction: Literal["minimize", "maximize"]
    max_features: int
    max_fits: int
    min_improvement: float = 0.0
    protected_features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_options(self.max_features, self.protected_features)
        _validate_options(self.max_fits, ())
        _validate_objective(self.metric, self.direction, self.min_improvement)

    def select(
        self,
        *,
        train: Pool,
        validation: Pool,
        params: ModelParams,
        model_factory: Callable[[ModelParams], Model],
        metric: Metric,
    ) -> FeatureSelectionResult:
        """Return a selected subset and its complete search history.

        Parameters
        ----------
        train, validation : Pool
            Aligned prepared data with identical feature names and order.
        params : ModelParams
            Base constructor parameters; copy before passing to a fresh model.
        model_factory : callable
            Constructs an independent model for every fit.
        metric : Metric
            Evaluates targets, predictions and weights on validation rows.

        Returns
        -------
        FeatureSelectionResult
            Nonempty unique input features in original order, their selected
            history index, all evaluated subsets and serializable settings.
            Scores guide selection and are not independent quality estimates.
        """
        _validate_data(train, validation, self.max_features, self.protected_features)
        names = tuple(train.X.columns)
        selected = tuple(n for n in names if n in self.protected_features)
        initial_cost = int(bool(selected))
        if len(selected) < self.max_features:
            initial_cost += len(names) - len(selected)
        if self.max_fits < initial_cost:
            raise ValueError(
                "max_fits is too small for the initial complete comparison"
            )
        history: list[FeatureSelectionStep] = []
        selected_index = -1
        if selected:
            history.append(
                _evaluate(
                    selected,
                    train,
                    validation,
                    params,
                    model_factory,
                    metric,
                    self.metric,
                )
            )
            selected_index = 0
        while len(selected) < self.max_features:
            remaining = [n for n in names if n not in selected]
            if len(history) + len(remaining) > self.max_fits:
                break
            first = len(history)
            for candidate in remaining:
                subset = tuple(n for n in names if n in selected or n == candidate)
                history.append(
                    _evaluate(
                        subset,
                        train,
                        validation,
                        params,
                        model_factory,
                        metric,
                        self.metric,
                    )
                )
            choose = min if self.direction == "minimize" else max
            best_index = choose(
                range(first, len(history)),
                key=lambda i: history[i].metrics[self.metric],
            )
            if selected_index >= 0:
                previous = history[selected_index].metrics[self.metric]
                best = history[best_index].metrics[self.metric]
                improvement = (
                    previous - best if self.direction == "minimize" else best - previous
                )
                if improvement <= self.min_improvement:
                    break
            selected_index = best_index
            selected = history[selected_index].features
        return FeatureSelectionResult(
            selected,
            selected_index,
            tuple(history),
            {
                "selector": "forward",
                "metric": self.metric,
                "direction": self.direction,
                "max_features": self.max_features,
                "max_fits": self.max_fits,
                "min_improvement": self.min_improvement,
                "protected_features": self.protected_features,
            },
        )
