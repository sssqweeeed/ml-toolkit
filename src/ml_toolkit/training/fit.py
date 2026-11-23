"""The complete train/validation unit and last-split final refitting."""

from collections.abc import Callable
from dataclasses import dataclass, replace

import mlflow
from optuna import Trial

from .._console import Logger, capture_messages, message
from ..config import TrainingConfig, TuningConfig
from ..data import Pool, validate_compatible_pools, validate_training_categories
from ..feature_selection import FeatureSelectionResult, FeatureSelector
from ..metrics import Metric
from ..models import TrackingConfig
from ..models.base import Model
from ..reports import DataAnalyzer, ModelEvaluator, PredictionEvaluator
from ..tracking import log_json, nested_run
from ..types import ModelParams
from .evaluation import evaluate_model
from .reporting import (
    log_model_with_baseline,
    prepare_psi_baseline,
    print_datasets,
    report_feature_selection,
)
from .results import SplitResult, TuningResult
from .tuning import tune_optuna

logger = Logger(__name__)


@dataclass(frozen=True)
class TuningPlan:
    """Enabled tuning: its settings together with the trial search space."""

    config: TuningConfig
    param_space: Callable[[Trial], ModelParams]


@dataclass(frozen=True)
class RunSettings:
    """Inputs shared by every fit of one training run."""

    model_factory: Callable[[ModelParams], Model]
    metric: Metric
    config: TrainingConfig
    evaluators: tuple[PredictionEvaluator, ...]
    analyzers: tuple[DataAnalyzer, ...]
    model_evaluators: tuple[ModelEvaluator, ...]


@dataclass(frozen=True)
class _SplitData:
    train: Pool
    validation: Pool
    index: int

    def select_features(self, names: tuple[str, ...]) -> "_SplitData":
        return _SplitData(
            self.train.select_features(names),
            self.validation.select_features(names),
            self.index,
        )


def fit_split(
    *,
    train: Pool,
    validation: Pool,
    run: RunSettings,
    params: ModelParams,
    feature_selector: FeatureSelector | None,
    tuning: TuningPlan | None,
    index: int,
) -> SplitResult:
    """Select features, tune, fit and report one validation split."""
    validate_training_categories(train)
    validate_training_categories(validation)
    # Public entry for direct callers; Split already guarantees this for the workflow.
    validate_compatible_pools(train, validation, context="Train and validation pools")
    logger.info("Starting validation split %s", index)
    logger.debug(
        "Split %s: %s training rows, %s validation rows",
        index,
        len(train),
        len(validation),
        console=False,
    )
    print_datasets(
        f"Validation split {index} data", {"train": train, "validation": validation}
    )
    data = _SplitData(train, validation, index)
    with nested_run(f"validation/split_{index}"):
        selection = None
        if feature_selector is not None:
            selection = _select_features(data, run, params, feature_selector)
            data = data.select_features(selection.features)
        tuned = None if tuning is None else _tune(data, run, params, tuning)
        selected_params = params.copy() if tuned is None else tuned.best_params.copy()
        mlflow.log_params(selected_params)
        result = _fit_selected(data, run, selected_params, selection, tuned)
    message(
        "Split %s: %s selected features; best iteration (zero-based): %s",
        index,
        len(result.features),
        "—" if result.best_iteration is None else result.best_iteration,
        verbosity="verbose",
    )
    logger.info("Completed validation split %s", index)
    return result


def _select_features(
    data: _SplitData,
    run: RunSettings,
    params: ModelParams,
    selector: FeatureSelector,
) -> FeatureSelectionResult:
    """Run the selector, check its answer and log its history."""
    logger.info("Starting feature selection")
    with capture_messages() as selection_messages:
        selection = selector.select(
            train=data.train,
            validation=data.validation,
            params=params.copy(),
            model_factory=run.model_factory,
            metric=run.metric,
        )
    names = selection.features
    if (
        not names
        or len(set(names)) != len(names)
        or tuple(n for n in data.train.X.columns if n in names) != names
    ):
        raise ValueError("Selector must return unique input features in original order")
    if (
        not 0 <= selection.selected_step < len(selection.steps)
        or selection.steps[selection.selected_step].features != names
    ):
        raise ValueError("Selector selected_step must identify its returned features")
    summary = (
        f"Completed feature selection: {len(names)} features selected "
        f"at step {selection.selected_step + 1}"
    )
    if summary not in selection_messages:
        message(summary)
    features = f"Selected features: {names}"
    if features not in selection_messages:
        message(features, verbosity="debug")
    log_json(selection.to_dict(), "selection/history.json")
    selected = selection.steps[selection.selected_step]
    mlflow.log_metrics(
        {
            "selection/n_fits": float(len(selection.steps)),
            "selection/n_features": float(len(names)),
            "selection/selected_step": float(selection.selected_step),
            **{
                f"selection/metrics/{key}": value
                for key, value in selected.metrics.items()
            },
            **{
                f"selection/objective_metrics/{key}": value
                for key, value in (selected.objective_metrics or {}).items()
            },
        }
    )
    report_feature_selection(
        selection, run.config.reporting, stage=f"validation/split_{data.index}"
    )
    return selection


@dataclass(frozen=True)
class _TrialObjective:
    """Optuna objective: fit, score, report and optionally log one candidate."""

    data: _SplitData
    run: RunSettings
    tuning: TuningConfig

    def __call__(self, candidate: ModelParams) -> float:
        run, data, tuning = self.run, self.data, self.tuning
        config = run.config
        trial_model = run.model_factory(candidate.copy()).fit(
            data.train, data.validation
        )
        baseline = prepare_psi_baseline(
            data.train,
            config.reporting,
            log_model=config.tracking.log_trial_models,
            stage="tuning",
        )
        try:
            trial_scores = evaluate_model(
                trial_model,
                data.train,
                data.validation,
                run.metric,
                config.reporting,
                run.evaluators,
                analyzers=run.analyzers,
                model_evaluators=run.model_evaluators,
                psi_baseline=baseline,
                stage="tuning",
                name=f"validation/split_{data.index}/tuning",
                objective_evaluator=tuning.evaluator,
            )
        except Exception as exc:
            if tuning.evaluator is not None:
                exc.add_note(
                    f"Tuning evaluator={tuning.evaluator.name}, "
                    f"metric={tuning.metric!r}, split={data.index}"
                )
            raise
        if config.tracking.log_trial_models:
            log_model_with_baseline(
                trial_model,
                data.validation,
                _model_tracking(config),
                baseline,
            )
        if tuning.evaluator is None:
            return (trial_scores.heldout_metrics or {})[tuning.metric]
        objective_metrics = trial_scores.objective_metrics or {}
        if tuning.metric not in objective_metrics:
            raise ValueError(
                f"Tuning evaluator={tuning.evaluator.name}, "
                f"metric={tuning.metric!r}, split={data.index}: "
                f"objective key is missing; available keys: {sorted(objective_metrics)}"
            )
        return objective_metrics[tuning.metric]


def _model_tracking(config: TrainingConfig) -> TrackingConfig:
    """Return tracking settings for models that are never registered."""
    return replace(config.tracking, registered_model_name=None)


def _tune(
    data: _SplitData, run: RunSettings, params: ModelParams, plan: TuningPlan
) -> TuningResult:
    return tune_optuna(
        base_params=params,
        param_space=plan.param_space,
        score=_TrialObjective(data, run, plan.config),
        config=plan.config,
    )


def _fit_selected(
    data: _SplitData,
    run: RunSettings,
    selected_params: ModelParams,
    selection: FeatureSelectionResult | None,
    tuned: TuningResult | None,
) -> SplitResult:
    """Fit the split's final model, report it and build the split result."""
    config = run.config
    model = run.model_factory(selected_params.copy()).fit(data.train, data.validation)
    baseline = prepare_psi_baseline(
        data.train,
        config.reporting,
        log_model=config.tracking.log_validation_models,
        stage="validation",
    )
    scores = evaluate_model(
        model,
        data.train,
        data.validation,
        run.metric,
        config.reporting,
        run.evaluators,
        analyzers=run.analyzers,
        model_evaluators=run.model_evaluators,
        psi_baseline=baseline,
        stage="validation",
        name=f"validation/split_{data.index}",
    )
    if config.tracking.log_validation_models:
        log_model_with_baseline(
            model, data.validation, _model_tracking(config), baseline
        )
    return SplitResult(
        index=data.index,
        train_metrics=scores.train_metrics,
        val_metrics=scores.heldout_metrics or {},
        features=tuple(data.train.X.columns),
        refit_params=model.get_refit_params(),
        best_iteration=model.best_iteration,
        feature_selection=selection,
        tuning=tuned,
    )


def refit_model(
    *,
    data: Pool,
    last_split: SplitResult,
    model_factory: Callable[[ModelParams], Model],
) -> tuple[Model, ModelParams]:
    validate_training_categories(data)
    params = last_split.refit_params.copy()
    logger.info("Refitting model without validation")
    mlflow.log_params(params)
    logger.debug("Refit selected iterations: %s", last_split.best_iteration)
    model = model_factory(params.copy()).fit(data)
    return model, params
