"""Batch RFE through model adapters, with a fixed chronological holdout."""

import logging
from collections.abc import Callable, Mapping
from numbers import Real
from time import perf_counter

import numpy as np

from ..config import RFEFeatureSelectionConfig
from ..models.base import Model
from .contracts import FeatureRanker, TrainingData, validate_training_data
from .evaluation import _scores, _validate_task_metric
from .results import FeatureSelectionResult, FeatureSelectionStep

logger = logging.getLogger(__name__)


def _selection_data(data, config):
    validate_training_data(data)
    names = data.X.columns
    if config.min_features > len(names):
        raise ValueError("min_features must not exceed the input feature count")
    missing = set(config.protected_features) - set(names)
    if missing:
        raise ValueError(f"Protected features are missing from X: {sorted(missing)}")
    split = int(len(data) * (1 - config.validation_fraction))
    train_end = split - config.gap
    if train_end <= 0 or split >= len(data):
        raise ValueError(
            "Not enough rows for feature-selection train, gap and validation"
        )
    train = data.take(np.arange(train_end))
    validation = data.take(np.arange(split, len(data)))
    validate_training_data(train)
    validate_training_data(validation)
    return train, validation


def _feature_counts(count, config):
    if count == config.min_features or config.max_fits == 1:
        return (count,)
    steps = min(config.max_fits, count - config.min_features + 1)
    sizes = np.rint(np.geomspace(count, config.min_features, steps)).astype(int)
    return tuple(dict.fromkeys(int(size) for size in sizes))


def _importances(ranker, model, data):
    values = ranker(model, data)
    if not isinstance(values, Mapping) or set(values) != set(data.X.columns):
        raise ValueError(
            "Feature ranker must return exactly one importance per input column"
        )
    result = {}
    for name in data.X.columns:
        value = values[name]
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not np.isfinite(value)
        ):
            raise ValueError(f"Feature importance for {name!r} must be a finite number")
        result[name] = float(value)
    return result


def select_features(
    *,
    data: TrainingData,
    params: dict,
    model_factory: Callable[[dict], Model],
    metric: Callable,
    ranker: FeatureRanker,
    config: RFEFeatureSelectionConfig,
) -> FeatureSelectionResult:
    """Fit shrinking feature subsets and retain the best acceptable subset.

    Parameters
    ----------
    data : TrainingData
        Chronological training-only rows. The caller excludes outer validation
        and test periods. Row operations must preserve weights and metadata.
    params : dict
        Fixed model settings copied for each new adapter; not mutated.
    model_factory : Callable[[dict], Model]
        Creates a fresh adapter, including fresh nested models and strategies.
    metric : Callable
        Existing target/prediction/weight metric returning a dataclass or mapping.
        config.metric_name must select a finite scalar in every step.
    ranker : FeatureRanker
        Called with the fitted adapter and internal validation subset. Returns
        finite values for exactly its feature names; larger means more important.
        Must not mutate the adapter or data. Not called after the final fit.
    config : RFEFeatureSelectionConfig
        Geometric fit budget, internal split, objective and protected columns.

    Returns
    -------
    FeatureSelectionResult
        Smallest visited subset within absolute tolerance of the best score,
        plus every step's metrics, ranking and timings. Contains no fitted models.

    Raises
    ------
    ValueError
        Data, protected names, split sizes, scores or ranking values are invalid.
    TypeError
        Configuration or executing dependencies have incorrect types.

    Notes
    -----
    Uses only adapter fit and predict_pool; never native feature selection.
    The same internal holdout may drive adapter early stopping and subset choice.
    Scores are search results, not an independent performance estimate. Does not
    start MLflow runs, log models, run diagnostics or invoke post-fit evaluators.
    Equal ranking values remove later columns first, retaining original order.

    Examples
    --------
    >>> from ml_toolkit.config import RFEFeatureSelectionConfig
    >>> settings = RFEFeatureSelectionConfig(
    ...     min_features=5, metric_name="mae", direction="minimize")
    >>> result = select_features(  # doctest: +SKIP
    ...     data=training_data, params=params, model_factory=factory,
    ...     metric=metric, ranker=ranker, config=settings)
    """
    if not isinstance(config, RFEFeatureSelectionConfig):
        raise TypeError("config must be an RFEFeatureSelectionConfig")
    if not all(callable(item) for item in (model_factory, metric, ranker)):
        raise TypeError("model_factory, metric and ranker must be callable")
    train, validation = _selection_data(data, config)
    counts = _feature_counts(len(data.X.columns), config)
    names = tuple(data.X.columns)
    protected = set(config.protected_features)
    history = []
    logger.info(
        "Starting feature selection: up to %s fits, minimum %s features",
        config.max_fits, config.min_features,
    )
    for index, count in enumerate(counts):
        step_train = train.select_features(names)
        step_validation = validation.select_features(names)
        started = perf_counter()
        model = model_factory(params.copy()).fit(step_train, step_validation)
        fit_seconds = perf_counter() - started
        _validate_task_metric(model, metric)
        started = perf_counter()
        scores = _scores(metric, step_validation, model.predict_pool(step_validation))
        if config.metric_name not in scores:
            raise ValueError(
                f"Metric {config.metric_name!r} is unavailable. "
                f"Available metrics: {sorted(scores)}"
            )
        if not np.isfinite(scores[config.metric_name]):
            raise ValueError(f"Selection metric {config.metric_name!r} must be finite")
        evaluation_seconds = perf_counter() - started
        importances, eliminated, ranking_seconds = {}, (), 0.0
        if index + 1 < len(counts):
            started = perf_counter()
            importances = _importances(ranker, model, step_validation)
            ranked = sorted(
                (name for name in names if name not in protected),
                key=importances.__getitem__, reverse=True,
            )
            retained = protected | set(ranked[: counts[index + 1] - len(protected)])
            eliminated = tuple(name for name in names if name not in retained)
            ranking_seconds = perf_counter() - started
        history.append(FeatureSelectionStep(
            names, scores, importances, eliminated, fit_seconds,
            evaluation_seconds, ranking_seconds, model.best_iteration,
        ))
        logger.info(
            "Feature selection step %s: %s features, %s=%s",
            index + 1, count, config.metric_name, scores[config.metric_name],
        )
        logger.debug("Eliminated features: %s", eliminated)
        if eliminated:
            names = tuple(name for name in names if name in retained)
        del model
    scores = [step.metrics[config.metric_name] for step in history]
    best = min(scores) if config.direction == "minimize" else max(scores)
    eligible = [
        index for index, score in enumerate(scores) if (
            score <= best + config.score_tolerance if config.direction == "minimize"
            else score >= best - config.score_tolerance
        )
    ]
    selected = min(eligible, key=lambda index: len(history[index].features))
    result = FeatureSelectionResult(history[selected].features, selected, tuple(history))
    logger.debug("Selected features: %s", result.features)
    logger.info(
        "Completed feature selection: %s features selected at step %s",
        len(result.features), selected + 1,
    )
    return result
