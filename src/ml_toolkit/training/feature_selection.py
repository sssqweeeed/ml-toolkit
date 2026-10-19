"""Batch RFE through model adapters on an explicit train/validation split."""

import logging
from collections.abc import Callable
from time import perf_counter

import numpy as np

from ..config import RFEFeatureSelectionConfig
from ..data import validate_training_data
from ..metrics import Metric
from ..models.base import Model
from ..types import ModelParams
from .contracts import FeatureRanker, Pool
from .evaluation import calculate_metrics
from .results import FeatureSelectionResult, FeatureSelectionStep

logger = logging.getLogger(__name__)


def _feature_counts(count: int, config: RFEFeatureSelectionConfig) -> tuple[int, ...]:
    if count == config.min_features or config.max_fits == 1:
        return (count,)
    steps = min(config.max_fits, count - config.min_features + 1)
    sizes = np.rint(np.geomspace(count, config.min_features, steps)).astype(int)
    return tuple(dict.fromkeys(int(size) for size in sizes))


def _importances(
    ranker: FeatureRanker, model: Model, data: Pool
) -> dict[str, float]:
    values = ranker(model, data)
    if set(values) != set(data.X.columns):
        raise ValueError(
            "Feature ranker must return exactly one importance per input column"
        )
    result: dict[str, float] = {}
    for name in data.X.columns:
        value = values[name]
        if not np.isfinite(value):
            raise ValueError(f"Feature importance for {name!r} must be a finite number")
        result[name] = float(value)
    return result


def select_features(
    *,
    train: Pool,
    validation: Pool,
    params: ModelParams,
    model_factory: Callable[[ModelParams], Model],
    metric: Metric,
    ranker: FeatureRanker,
    config: RFEFeatureSelectionConfig,
) -> FeatureSelectionResult:
    """Fit shrinking feature subsets and retain the best acceptable subset.

    Parameters
    ----------
    train, validation : Pool
        Explicit training and validation rows with matching feature columns.
        The caller excludes test. Row operations preserve weights and metadata.
    params : dict
        Fixed model settings copied for each new adapter; not mutated.
    model_factory : Callable[[ModelParams], Model]
        Creates a fresh adapter, including fresh nested models and strategies.
    metric : Callable
        Target/prediction/weight metric returning a result with to_dict().
        config.metric must select a finite scalar in every step.
    ranker : FeatureRanker
        Called with the fitted adapter and supplied validation subset. Returns
        finite values for exactly its feature names; larger means more important.
        Must not mutate the adapter or data. Not called after the final fit.
    config : RFEFeatureSelectionConfig
        Geometric fit budget, objective and protected columns.

    Returns
    -------
    FeatureSelectionResult
        Smallest visited subset within absolute tolerance of the best score,
        plus every step's metrics, ranking and timings. Contains no fitted models.

    Raises
    ------
    ValueError
        Data, feature alignment, protected names, scores or ranking are invalid.

    Notes
    -----
    Uses only adapter fit and predict_pool; never native feature selection.
    The supplied validation rows drive early stopping and subset choice.
    Scores are search results, not an independent performance estimate. Does not
    start MLflow runs, log models, run diagnostics or invoke post-fit evaluators.
    Equal ranking values remove later columns first, retaining original order.

    Examples
    --------
    >>> from ml_toolkit.config import RFEFeatureSelectionConfig
    >>> settings = RFEFeatureSelectionConfig(
    ...     min_features=5, metric="mae", direction="minimize")
    >>> result = select_features(  # doctest: +SKIP
    ...     train=train, validation=val, params=params, model_factory=factory,
    ...     metric=metric, ranker=ranker, config=settings)
    """
    validate_training_data(train)
    validate_training_data(validation)
    if train.X.columns != validation.X.columns:
        raise ValueError("Train and validation features must match")
    if config.min_features > len(train.X.columns):
        raise ValueError("min_features must not exceed the input feature count")
    missing = set(config.protected_features) - set(train.X.columns)
    if missing:
        raise ValueError(f"Protected features are missing from X: {sorted(missing)}")
    counts = _feature_counts(len(train.X.columns), config)
    names = tuple(train.X.columns)
    protected = set(config.protected_features)
    history: list[FeatureSelectionStep] = []
    logger.info(
        "Starting feature selection: up to %s fits, minimum %s features",
        config.max_fits,
        config.min_features,
    )
    for index, count in enumerate(counts):
        step_train = train.select_features(names)
        step_validation = validation.select_features(names)
        started = perf_counter()
        model = model_factory(params.copy()).fit(step_train, step_validation)
        fit_seconds = perf_counter() - started
        started = perf_counter()
        scores = calculate_metrics(
            metric, step_validation, model.predict_pool(step_validation)
        )
        if config.metric not in scores:
            raise ValueError(
                f"Metric {config.metric!r} is unavailable. "
                f"Available metrics: {sorted(scores)}"
            )
        if not np.isfinite(scores[config.metric]):
            raise ValueError(f"Selection metric {config.metric!r} must be finite")
        evaluation_seconds = perf_counter() - started
        importances: dict[str, float] = {}
        eliminated: tuple[str, ...] = ()
        retained = set(names)
        ranking_seconds = 0.0
        if index + 1 < len(counts):
            started = perf_counter()
            importances = _importances(ranker, model, step_validation)
            ranked = sorted(
                (name for name in names if name not in protected),
                key=importances.__getitem__,
                reverse=True,
            )
            retained = protected | set(ranked[: counts[index + 1] - len(protected)])
            eliminated = tuple(name for name in names if name not in retained)
            ranking_seconds = perf_counter() - started
        history.append(
            FeatureSelectionStep(
                names,
                scores,
                importances,
                eliminated,
                fit_seconds,
                evaluation_seconds,
                ranking_seconds,
                model.best_iteration,
            )
        )
        logger.info(
            "Feature selection step %s: %s features, %s=%s",
            index + 1,
            count,
            config.metric,
            scores[config.metric],
        )
        logger.debug("Eliminated features: %s", eliminated)
        if eliminated:
            names = tuple(name for name in names if name in retained)
        del model
    objective_scores = [step.metrics[config.metric] for step in history]
    best = (
        min(objective_scores)
        if config.direction == "minimize"
        else max(objective_scores)
    )
    eligible = [
        index
        for index, score in enumerate(objective_scores)
        if (
            score <= best + config.score_tolerance
            if config.direction == "minimize"
            else score >= best - config.score_tolerance
        )
    ]
    selected = min(eligible, key=lambda index: len(history[index].features))
    result = FeatureSelectionResult(
        history[selected].features, selected, tuple(history)
    )
    logger.debug("Selected features: %s", result.features)
    logger.info(
        "Completed feature selection: %s features selected at step %s",
        len(result.features),
        selected + 1,
    )
    return result
