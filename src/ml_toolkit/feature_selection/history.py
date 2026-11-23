"""Model-free feature selection histories and their typed search settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from itertools import groupby
from typing import Literal, TypeAlias

__all__ = [
    "FeatureSelectionResult",
    "FeatureSelectionStep",
    "RFESettings",
    "TopKSettings",
    "ForwardSettings",
    "CustomSelectionSettings",
    "SelectionSettings",
    "SettingValue",
    "ForwardRound",
    "ForwardStop",
    "forward_rounds",
    "forward_stop",
]

SettingValue: TypeAlias = str | int | float | bool | None | tuple[str, ...]
"""JSON-compatible scalar or feature-name tuple of a custom selector setting."""


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
    objective_metrics : dict[str, float] or None, default None
        Finite evaluator scores with their exact returned keys, separate from
        ordinary metrics. None means the ordinary metric supplies the objective.
    """

    features: tuple[str, ...]
    metrics: dict[str, float]
    importances: dict[str, float]
    eliminated_features: tuple[str, ...]
    fit_seconds: float
    evaluation_seconds: float
    ranking_seconds: float
    best_iteration: int | None
    objective_metrics: dict[str, float] | None = None

    def objective_score(self, metric: str) -> float:
        """Return the selection objective's value for this subset.

        Parameters
        ----------
        metric : str
            Evaluator key when ``objective_metrics`` is set, otherwise the
            ordinary metric key.

        Returns
        -------
        float
            Stored objective score.

        Raises
        ------
        KeyError
            The step does not store the requested score.
        """
        scores = (
            self.metrics if self.objective_metrics is None else self.objective_metrics
        )
        return scores[metric]


@dataclass(frozen=True, kw_only=True)
class RFESettings:
    """Record the configuration of a recursive feature elimination search.

    Attributes
    ----------
    selector : Literal['rfe']
        Selector identifier stored in the logged history.
    min_features, max_fits : int
        Requested minimum subset size and the fit budget.
    metric : str
        Objective key.
    direction : Literal['minimize', 'maximize']
        Objective direction.
    score_tolerance : float
        Absolute distance from the best score within which the smallest subset
        is chosen.
    protected_features : tuple[str, ...]
        Columns retained throughout.
    evaluator : str or None, default None
        Name of the objective evaluator; omitted from logged JSON when None.
    """

    selector: Literal["rfe"] = field(default="rfe", init=False)
    min_features: int
    metric: str
    direction: Literal["minimize", "maximize"]
    max_fits: int
    score_tolerance: float
    protected_features: tuple[str, ...]
    evaluator: str | None = None


@dataclass(frozen=True, kw_only=True)
class TopKSettings:
    """Record the configuration of a Top-K ranking search.

    Attributes
    ----------
    selector : Literal['top_k']
        Selector identifier stored in the logged history.
    k : int
        Selected feature count, including protected features.
    protected_features : tuple[str, ...]
        Mandatory columns counted within k.
    """

    selector: Literal["top_k"] = field(default="top_k", init=False)
    k: int
    protected_features: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class ForwardSettings:
    """Record the configuration of a forward selection search.

    Attributes
    ----------
    selector : Literal['forward']
        Selector identifier stored in the logged history.
    metric : str
        Objective key.
    direction : Literal['minimize', 'maximize']
        Objective direction.
    max_features, max_fits : int
        Result size and total fit budget.
    min_improvement : float
        Strict minimum absolute improvement required to accept an addition.
    protected_features : tuple[str, ...]
        Mandatory starting subset.
    evaluator : str or None, default None
        Name of the objective evaluator; omitted from logged JSON when None.
    """

    selector: Literal["forward"] = field(default="forward", init=False)
    metric: str
    direction: Literal["minimize", "maximize"]
    max_features: int
    max_fits: int
    min_improvement: float
    protected_features: tuple[str, ...]
    evaluator: str | None = None


@dataclass(frozen=True)
class CustomSelectionSettings:
    """Record the configuration of a project-owned selector.

    Attributes
    ----------
    selector : str
        Selector identifier. Built-in plotting skips identifiers it does not
        know.
    details : dict[str, SettingValue]
        Concrete configuration values; flattened next to ``selector`` in the
        logged history. Never store models, rankers or evaluator objects.
    """

    selector: str
    details: dict[str, SettingValue] = field(default_factory=lambda: {})


SelectionSettings: TypeAlias = (
    RFESettings | TopKSettings | ForwardSettings | CustomSelectionSettings
)
"""Typed, JSON-compatible selector configuration, without executable objects."""


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
    settings : SelectionSettings or None, default None
        Concrete configuration; excludes models, rankers and evaluator objects.
        Evaluator-based selectors record the evaluator name. None means no
        configuration is recorded.

    Notes
    -----
    Validation scores describe selection, not independent performance.
    Frozen result objects contain mutable metric and importance dictionaries.
    """

    features: tuple[str, ...]
    selected_step: int
    steps: tuple[FeatureSelectionStep, ...]
    settings: SelectionSettings | None = None

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-compatible payload logged as ``selection/history.json``.

        Returns
        -------
        dict[str, object]
            Fields of the result; ``settings`` is a flat mapping of the
            selector's configuration without unset evaluator names.
        """
        payload: dict[str, object] = asdict(replace(self, settings=None))
        payload["settings"] = _settings_payload(self.settings)
        return payload


def _settings_payload(settings: SelectionSettings | None) -> dict[str, SettingValue]:
    if settings is None:
        return {}
    if isinstance(settings, CustomSelectionSettings):
        return {"selector": settings.selector, **settings.details}
    return {
        key: value
        for key, value in asdict(settings).items()
        if not (key == "evaluator" and value is None)
    }


@dataclass(frozen=True)
class ForwardRound:
    """Candidate subsets of one size in a forward search.

    Attributes
    ----------
    count : int
        Feature count of every candidate in the round.
    indices : tuple[int, ...]
        Step indices of the round's candidates.
    best : int
        Step index of the round's best candidate (the first one for a
        protected baseline).
    accepted : bool
        Whether the round's best candidate lies within the selected subset.
    baseline : bool
        Whether the round is the protected-only starting fit.
    """

    count: int
    indices: tuple[int, ...]
    best: int
    accepted: bool
    baseline: bool


def forward_rounds(
    result: FeatureSelectionResult, settings: ForwardSettings
) -> list[ForwardRound]:
    """Split a forward history into candidate rounds and pick each winner.

    Parameters
    ----------
    result : FeatureSelectionResult
        Forward history whose steps store the objective named by settings.
    settings : ForwardSettings
        Objective and protected features of that history.

    Returns
    -------
    list[ForwardRound]
        Consecutive equal-size candidates with their best step.

    Raises
    ------
    ValueError
        The history does not identify the selected step as its last accepted
        round winner.
    """
    maximize = settings.direction == "maximize"
    scores = [step.objective_score(settings.metric) for step in result.steps]
    protected = settings.protected_features
    rounds: list[ForwardRound] = []
    for size, group in groupby(
        range(len(result.steps)), key=lambda i: len(result.steps[i].features)
    ):
        indices = tuple(group)
        baseline = bool(protected) and not rounds and size == len(protected)
        best = (
            indices[0]
            if baseline
            else (max if maximize else min)(indices, key=lambda i: scores[i])
        )
        rounds.append(
            ForwardRound(size, indices, best, size <= len(result.features), baseline)
        )
    accepted = [item.best for item in rounds if item.accepted]
    if not accepted or accepted[-1] != result.selected_step:
        raise ValueError("Forward history must identify its accepted round winners")
    return rounds


@dataclass(frozen=True)
class ForwardStop:
    """Why a forward search ended.

    Attributes
    ----------
    reason : Literal['improvement', 'max_features', 'max_fits']
        The rejected best candidate's gain was too small, the size limit was
        reached, or the next full round did not fit the fit budget.
    rejected_step : int or None
        Step index of the best candidate of the rejected round, for the
        ``improvement`` reason.
    """

    reason: Literal["improvement", "max_features", "max_fits"]
    rejected_step: int | None = None


def forward_stop(
    result: FeatureSelectionResult,
    settings: ForwardSettings,
    rounds: list[ForwardRound],
) -> ForwardStop:
    """Return the stopping reason of a forward history.

    Parameters
    ----------
    result : FeatureSelectionResult
        Forward history.
    settings : ForwardSettings
        Its configuration.
    rounds : list[ForwardRound]
        Rounds returned by ``forward_rounds`` for the same history.

    Returns
    -------
    ForwardStop
        Reason and, when a round was rejected, its best step.
    """
    rejected = [item.best for item in rounds if not item.accepted]
    if rejected:
        return ForwardStop("improvement", rejected[-1])
    if len(result.features) >= settings.max_features:
        return ForwardStop("max_features")
    return ForwardStop("max_fits")
