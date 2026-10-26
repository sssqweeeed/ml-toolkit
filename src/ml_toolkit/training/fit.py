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
from .contracts import Pool
from ..feature_selection import FeatureSelector
from ..data import validate_training_categories
from .evaluation import evaluate_model
from .psi import prepare_psi_baseline, log_model_with_baseline
from ..evaluation import PredictionEvaluator
from ..data_analysis import DataAnalyzer
from .results import SplitResult
from .tuning import tune_optuna

logger = logging.getLogger(__name__)


def fit_split(
    *,
    train: Pool,
    validation: Pool,
    params: ModelParams,
    model_factory: Callable[[ModelParams], Model],
    metric: Metric,
    config: TrainingConfig,
    feature_selector: FeatureSelector | None,
    evaluators: tuple[PredictionEvaluator, ...],
    analyzers: tuple[DataAnalyzer, ...],
    param_space: Callable[[Trial], ModelParams] | None,
    index: int,
) -> SplitResult:
    validate_training_categories(train)
    validate_training_categories(validation)
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
        if feature_selector is not None:
            selection = feature_selector.select(
                train=train,
                validation=validation,
                params=params.copy(),
                model_factory=model_factory,
                metric=metric,
            )
            names = selection.features
            if (
                not names
                or len(set(names)) != len(names)
                or tuple(n for n in train.X.columns if n in names) != names
            ):
                raise ValueError(
                    "Selector must return unique input features in original order"
                )
            if (
                not 0 <= selection.selected_step < len(selection.steps)
                or selection.steps[selection.selected_step].features != names
            ):
                raise ValueError(
                    "Selector selected_step must identify its returned features"
                )
            train = train.select_features(names)
            validation = validation.select_features(names)
            tracking.log_json(asdict(selection), "selection/history.json")
            tracking.log_metrics(
                {
                    "selection/n_fits": float(len(selection.steps)),
                    "selection/n_features": float(len(names)),
                    "selection/selected_step": float(selection.selected_step),
                    **{
                        f"selection/metrics/{key}": value
                        for key, value in selection.steps[
                            selection.selected_step
                        ].metrics.items()
                    },
                }
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
                baseline = prepare_psi_baseline(
                    train,
                    config.reporting,
                    log_model=config.tracking.log_trial_models,
                    stage="tuning",
                )
                trial_scores = evaluate_model(
                    trial_model,
                    train,
                    validation,
                    metric,
                    config.reporting,
                    evaluators,
                    analyzers=analyzers,
                    psi_baseline=baseline,
                    stage="tuning",
                    name=f"validation/split_{index}/tuning",
                )
                if config.tracking.log_trial_models:
                    log_model_with_baseline(
                        trial_model, validation, model_tracking, baseline
                    )
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
        baseline = prepare_psi_baseline(
            train,
            config.reporting,
            log_model=config.tracking.log_validation_models,
            stage="validation",
        )
        scores = evaluate_model(
            model,
            train,
            validation,
            metric,
            config.reporting,
            evaluators,
            analyzers=analyzers,
            psi_baseline=baseline,
            stage="validation",
            name=f"validation/split_{index}",
        )
        if config.tracking.log_validation_models:
            log_model_with_baseline(model, validation, model_tracking, baseline)
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


def refit_model(
    *,
    data: Pool,
    last_split: SplitResult,
    model_factory: Callable[[ModelParams], Model],
) -> tuple[Model, ModelParams]:
    validate_training_categories(data)
    params = last_split.refit_params.copy()
    logger.info("Refitting model without validation")
    tracking.log_params(params)
    logger.debug("Refit selected iterations: %s", last_split.best_iteration)
    model = model_factory(params.copy()).fit(data)
    return model, params
