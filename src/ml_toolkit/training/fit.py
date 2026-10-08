"""The complete train/validation unit and last-split final refitting."""

import logging
from collections.abc import Callable
from dataclasses import asdict, replace

from optuna import Trial

from ..config import TrainingConfig
from ..metrics import Metric
from ..models.base import Model
from ..tracking import runs as tracking
from ..types import ModelParams
from .contracts import FeatureRanker, PostFitEvaluator, TrainingData
from .evaluation import evaluate_model
from .feature_selection import select_features
from .results import SplitResult
from .tuning import tune_optuna

logger = logging.getLogger(__name__)


def fit_split(
    *,
    train: TrainingData,
    validation: TrainingData,
    params: ModelParams,
    model_factory: Callable[[ModelParams], Model],
    metric: Metric,
    config: TrainingConfig,
    feature_ranker: FeatureRanker | None,
    post_fit_evaluator: PostFitEvaluator | None,
    param_space: Callable[[Trial], ModelParams] | None,
    index: int,
    diagnostics: bool,
) -> SplitResult:
    if train.X.columns != validation.X.columns:
        raise ValueError("Train and validation features must match")
    logger.info("Starting validation split %s", index)
    logger.debug(
        "Split %s: %s training rows, %s validation rows",
        index,
        len(train),
        len(validation),
    )
    with tracking.run(f"validation/split_{index}"):
        selection = None
        if config.feature_selection is not None:
            if feature_ranker is None:
                raise ValueError(
                    "feature_ranker is required when feature selection is enabled"
                )
            selection = select_features(
                train=train,
                validation=validation,
                params=params,
                model_factory=model_factory,
                metric=metric,
                ranker=feature_ranker,
                config=config.feature_selection,
            )
            train = train.select_features(selection.features)
            validation = validation.select_features(selection.features)
            tracking.log_json(asdict(selection), "selection/history.json")
            selected_score = selection.steps[selection.selected_step].metrics[
                config.feature_selection.metric
            ]
            tracking.log_metrics(
                {
                    "selection/n_fits": float(len(selection.steps)),
                    "selection/n_features": float(len(selection.features)),
                    "selection/selected_step": float(selection.selected_step),
                    "selection/score": selected_score,
                },
            )
        tuning = None
        selected_params = params.copy()
        model_tracking = replace(config.tracking, registered_model_name=None)
        if config.tuning is not None:
            if param_space is None:
                raise ValueError("param_space is required when tuning is enabled")
            tuning_config = config.tuning

            def score(candidate: ModelParams) -> float:
                trial_model = model_factory(candidate.copy()).fit(train, validation)
                trial_scores = evaluate_model(
                    trial_model,
                    train,
                    validation,
                    metric,
                    config.reporting,
                    post_fit_evaluator,
                    stage="tuning",
                    name=f"validation/split_{index}/tuning",
                )
                if config.tracking.log_trial_models:
                    trial_model.log(validation, config=model_tracking)
                return (trial_scores.heldout_metrics or {})[tuning_config.metric]

            tuning = tune_optuna(
                base_params=params,
                param_space=param_space,
                score=score,
                config=tuning_config,
            )
            selected_params = tuning.best_params.copy()
        tracking.log_params(selected_params)
        model = model_factory(selected_params.copy()).fit(train, validation)
        scores = evaluate_model(
            model,
            train,
            validation,
            metric,
            config.reporting,
            post_fit_evaluator,
            stage="validation",
            name=f"validation/split_{index}",
            diagnostics=diagnostics,
        )
        if config.tracking.log_validation_models:
            model.log(validation, config=model_tracking)
        result = SplitResult(
            index=index,
            train_metrics=scores.train_metrics,
            val_metrics=scores.heldout_metrics or {},
            features=tuple(train.X.columns),
            refit_params=model.get_refit_params(),
            best_iteration=model.best_iteration,
            feature_selection=selection,
            tuning=tuning,
        )
    logger.info("Completed validation split %s", index)
    return result


def fit_final_model(
    *,
    data: TrainingData,
    last_split: SplitResult,
    model_factory: Callable[[ModelParams], Model],
) -> tuple[Model, TrainingData, ModelParams]:
    params = last_split.refit_params.copy()
    train = data.select_features(last_split.features)
    logger.info("Starting final fit")
    tracking.log_params(params)
    logger.debug("Final fit selected iterations: %s", last_split.best_iteration)
    model = model_factory(params.copy()).fit(train)
    return model, train, params
