"""Selection, optimization, validation and final/test orchestration of a run."""

from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, replace
from typing import Literal

import mlflow
from optuna import Trial

from .._console import Logger, message, metrics, output_context
from ..config import TrainingConfig
from ..config.training import check_registration_target_logged
from ..data import Pool, PoolSchema, validate_compatible_pools, validate_training_data
from ..feature_selection import FeatureSelector
from ..metrics import Metric
from ..models import TrackingConfig
from ..models.base import Model
from ..reports import DataAnalyzer, ModelEvaluator, PredictionEvaluator
from ..tracking import log_json, nested_run
from ..types import ModelParams
from .evaluation import EvaluationResult, evaluate_model
from .fit import RunSettings, TuningPlan, fit_split, refit_model
from .reporting import (
    log_model_with_baseline,
    prepare_analyzers,
    prepare_evaluators,
    prepare_model_evaluators,
    prepare_psi_baseline,
    print_datasets,
)
from .results import DatasetSplit, SplitResult, TrainingResult, ValidationResult
from .splits import Split, Splitter

logger = Logger(__name__)


def train_model(
    *,
    splitter: Splitter[Pool],
    params: ModelParams,
    model_factory: Callable[[ModelParams], Model],
    metric: Metric,
    run_name: str,
    config: TrainingConfig,
    feature_selector: FeatureSelector | None = None,
    param_space: Callable[[Trial], ModelParams] | None = None,
    prediction_evaluators: (
        PredictionEvaluator | list[PredictionEvaluator] | None
    ) = None,
    data_analyzers: DataAnalyzer | list[DataAnalyzer] | None = None,
    model_evaluators: ModelEvaluator | list[ModelEvaluator] | None = None,
) -> TrainingResult:
    """Tune, validate, refit and optionally test a complete training workflow.

    Parameters
    ----------
    splitter : Splitter
        Supplies numbered validation pairs and final training/test data. Pools
        must have aligned features and metadata, ascending timestamps, finite
        targets and nonnegative weights with positive total. Raw categorical
        values must not stringify to the reserved "unknown".
    params : ModelParams
        Base constructor parameters, copied without mutating the input.
    model_factory : Callable[[ModelParams], Model]
        Creates fresh adapters implementing fit, predict_pool, native_model,
        best_iteration, feature_importances, get_refit_params and log.
    metric : Callable
        Receives targets, predictions and weights; returns named scalar scores.
    run_name : str
        Enclosing MLflow experiment run name.
    config : TrainingConfig
        Parameter tuning, console verbosity, reporting and model packaging settings.
    feature_selector : FeatureSelector or None, default None
        Selects features independently before tuning on each split. None keeps
        all columns. The selector owns its settings and must not retain fitted state.
    param_space : Callable or None, default None
        Optuna trial overrides; required when tuning is enabled.
    prediction_evaluators : PredictionEvaluator, list[PredictionEvaluator], or None, default None
        Named evaluators called in list order with one combined pool and aligned
        train and optional val or test labels per fit. Names are unique ignoring
        case; object IDs are optional. ReportingConfig stage flags gate all reports.
        None or an empty list disables prediction reports, not data analysis or PSI/SHAP.
        An evaluator configured as a tuning objective runs on trials independently;
        include the same instance here to also enable its reports.
    data_analyzers : DataAnalyzer, list[DataAnalyzer], or None, default None
        Data reports receiving the same pool and split labels without predictions.
        Called in list order after prediction/model evaluators, using the same stage
        flags and selected features. Names must be unique across all report lists and
        enabled PSI/SHAP. None or [] disables custom data reports, not PSI.

    model_evaluators : ModelEvaluator, list[ModelEvaluator], or None, default None
        Reports receiving the fitted adapter, task metric and aligned data/split
        labels. Run after prediction evaluators and before data analyzers, under
        the same reporting stage flags. Names are unique across all reports.

    Returns
    -------
    TrainingResult
        Main model, split results, optional test scores and additional final model.

    Notes
    -----
    Each split independently selects features using the base parameters, tunes
    on its validation rows, then fits the winning settings once for reporting.
    Validation averages are reports, not Optuna objectives. The main model
    fits splitter.get_test().train using the last split's features and refit
    parameters. With held-out test data this is the test stage; otherwise it
    is final_fit. fit_on_test adds a fresh final_fit on splitter.get_final()
    after test evaluation, without replacing the main model or its scores.
    No selection or tuning is repeated. MLflow configuration is caller-owned.

    See Also
    --------
    ml_toolkit.training.Holdout : A trailing validation window and optional test.
    ml_toolkit.training.TimeSeriesCV : Chronological validation windows.

    Examples
    --------
    For executable data preparation, configuration, fitting and interpretation
    of held-out scores, see the documentation's Training guide, "Run a holdout
    experiment". The Feature selection, Parameter tuning and Evaluation guides
    show how to extend that workflow with independently prepared examples.
    """
    with output_context(config.verbosity):
        _check_registration_target(splitter, config)
        splits = _validation_splits(splitter)
        tuning = _tuning_plan(config, param_space)
        analyzers = prepare_analyzers(data_analyzers)
        model_reports = prepare_model_evaluators(model_evaluators)
        objective_evaluator = None if config.tuning is None else config.tuning.evaluator
        run = RunSettings(
            model_factory=model_factory,
            metric=metric,
            config=config,
            evaluators=prepare_evaluators(
                prediction_evaluators,
                config.reporting,
                analyzers,
                model_reports,
                objective_evaluator,
            ),
            analyzers=analyzers,
            model_evaluators=model_reports,
        )
        logger.info("Starting training run %s", run_name)
        message(
            "Dataset: %s rows, %s features",
            len(splitter.data),
            len(splitter.data.X.columns),
        )
        with nested_run(run_name):
            _log_configuration(splitter, run)
            validation = _run_validation(
                splits,
                run=run,
                params=params,
                feature_selector=feature_selector,
                tuning=tuning,
            )
            last = validation.last_split
            log_json(
                {
                    "features": last.features,
                    "model_params": last.refit_params,
                    "validation_split": last.index,
                },
                "configuration/resolved.json",
            )
            main = _run_main_stage(splitter, last, run)
            final_model = (
                _run_final_fit_stage(splitter, last, run, main.schema)
                if config.fit_on_test
                else None
            )
            _log_summary(validation.mean_metrics, main.result.scores.heldout_metrics)
            logger.info("Completed training run %s", run_name)
            return TrainingResult(
                model=main.result.model,
                validation=validation,
                params=main.result.params,
                features=last.features,
                feature_selection=last.feature_selection,
                tuning=last.tuning,
                test_metrics=main.result.scores.heldout_metrics,
                final_model=final_model,
            )


@dataclass(frozen=True)
class _StageResult:
    model: Model
    params: ModelParams
    scores: EvaluationResult


@dataclass(frozen=True)
class _StageSpec:
    stage: Literal["test", "final_fit"]
    title: str
    log_model: bool
    tracking: TrackingConfig


@dataclass(frozen=True)
class _MainStage:
    result: _StageResult
    # Only the feature layout is kept, so the main-stage pool can be freed before
    # the optional final fit.
    schema: PoolSchema


def _check_registration_target(
    splitter: Splitter[Pool], config: TrainingConfig
) -> None:
    tracking = config.tracking
    if (
        tracking.registered_model_name is not None
        and not config.fit_on_test
        and tracking.log_test_model != tracking.log_final_model
    ):
        # Only now does test-sample presence decide which flag matters.
        check_registration_target_logged(
            tracking,
            fit_on_test=False,
            has_test_sample=splitter.get_test().heldout is not None,
        )


def _validation_splits(splitter: Splitter[Pool]) -> Iterator[Split[Pool]]:
    """Return the validation iterator after checking that it yields a split."""
    splits = splitter.get_validation()
    first = next(splits, None)
    if first is None:
        raise ValueError("Splitter must yield at least one validation split")
    pending = [first]
    del first

    def stream() -> Iterator[Split[Pool]]:
        # The consumer owns each split, so it can release it before the next fit.
        yield pending.pop()
        yield from splits

    return stream()


def _tuning_plan(
    config: TrainingConfig, param_space: Callable[[Trial], ModelParams] | None
) -> TuningPlan | None:
    if config.tuning is None:
        return None
    if param_space is None:
        raise ValueError("param_space is required when tuning is enabled")
    return TuningPlan(config=config.tuning, param_space=param_space)


def _qualified_class(item: object) -> str:
    return f"{type(item).__module__}.{type(item).__qualname__}"


def _describe(item: PredictionEvaluator | DataAnalyzer | ModelEvaluator) -> object:
    return {
        "name": item.name,
        "class": _qualified_class(item),
        "parameters": item.parameters(),
    }


def _log_configuration(splitter: Splitter[Pool], run: RunSettings) -> None:
    config = run.config
    requested: dict[str, object] = asdict(replace(config, tuning=None))
    if config.tuning is not None:
        tuning_settings: dict[str, object] = asdict(
            replace(config.tuning, evaluator=None)
        )
        if config.tuning.evaluator is not None:
            tuning_settings["evaluator"] = _describe(config.tuning.evaluator)
        requested["tuning"] = tuning_settings
    log_json(requested, "configuration/requested.json")
    log_json(
        {"class": _qualified_class(splitter), "parameters": splitter.parameters()},
        "configuration/splitter.json",
    )
    log_json(
        [_describe(item) for item in run.evaluators],
        "configuration/evaluators.json",
    )
    log_json(
        [_describe(item) for item in run.analyzers],
        "configuration/data_analyzers.json",
    )
    log_json(
        [_describe(item) for item in run.model_evaluators],
        "configuration/model_evaluators.json",
    )


def _run_validation(
    splits: Iterator[Split[Pool]],
    *,
    run: RunSettings,
    params: ModelParams,
    feature_selector: FeatureSelector | None,
    tuning: TuningPlan | None,
) -> ValidationResult:
    """Fit every validation split and log the mean validation metrics."""
    logger.info("Starting validation")
    with nested_run("validation"):
        results: list[SplitResult] = []
        indices: set[int] = set()
        for part in splits:
            if part.heldout is None:
                raise ValueError("Validation split must contain heldout data")
            if part.index is None or part.index <= 0 or part.index in indices:
                raise ValueError("Validation split index must be positive and unique")
            indices.add(part.index)
            results.append(
                fit_split(
                    train=part.train,
                    validation=part.heldout,
                    run=run,
                    params=params,
                    feature_selector=feature_selector,
                    tuning=tuning,
                    index=part.index,
                )
            )
            del part
        validation = ValidationResult(tuple(results))
        mlflow.log_metrics(
            {
                f"validation/{key}": value
                for key, value in validation.mean_metrics.items()
            }
        )
    logger.info("Completed validation with %s splits", len(results))
    logger.debug("Validation mean metrics: %s", validation.mean_metrics, console=False)
    return validation


def _run_main_stage(
    splitter: Splitter[Pool], last: SplitResult, run: RunSettings
) -> _MainStage:
    """Refit on the non-test rows; test when a held-out sample exists."""
    config = run.config
    test = splitter.get_test()
    if config.fit_on_test and test.heldout is None:
        raise ValueError("fit_on_test=True requires a held-out test sample")
    train = test.train.select_features(last.features)
    heldout = (
        None if test.heldout is None else test.heldout.select_features(last.features)
    )
    del test
    stage: Literal["test", "final_fit"] = "final_fit" if heldout is None else "test"
    log_model = (
        config.tracking.log_final_model
        if heldout is None
        else config.tracking.log_test_model
    )
    # With fit_on_test the additional final_fit model is the registered one.
    tracking = (
        replace(config.tracking, registered_model_name=None)
        if config.fit_on_test
        else config.tracking
    )
    result = _run_stage(
        _StageSpec(stage, f"{stage} data", log_model, tracking),
        train=train,
        heldout=heldout,
        last=last,
        run=run,
    )
    return _MainStage(result, schema=PoolSchema.of(train))


def _run_final_fit_stage(
    splitter: Splitter[Pool], last: SplitResult, run: RunSettings, schema: PoolSchema
) -> Model:
    """Fit a fresh model on all eligible rows, including the test sample."""
    full = splitter.get_final()
    validate_training_data(full)
    full = full.select_features(last.features)
    validate_compatible_pools(schema, full, context="Final-fit data")
    spec = _StageSpec(
        "final_fit",
        "final_fit data (including test)",
        run.config.tracking.log_final_model,
        run.config.tracking,
    )
    return _run_stage(spec, train=full, heldout=None, last=last, run=run).model


def _run_stage(
    spec: _StageSpec,
    *,
    train: Pool,
    heldout: Pool | None,
    last: SplitResult,
    run: RunSettings,
) -> _StageResult:
    """Refit with the last split's settings, score, report and log the model."""
    stage, log_model = spec.stage, spec.log_model
    logger.info("Starting %s stage", stage)
    print_datasets(
        spec.title,
        {"train": train} if heldout is None else {"train": train, "test": heldout},
    )
    with nested_run(stage):
        model, params = refit_model(
            data=train, last_split=last, model_factory=run.model_factory
        )
        baseline = prepare_psi_baseline(
            train, run.config.reporting, log_model=log_model, stage=stage
        )
        scores = evaluate_model(
            model,
            train,
            heldout,
            run.metric,
            run.config.reporting,
            run.evaluators,
            analyzers=run.analyzers,
            model_evaluators=run.model_evaluators,
            psi_baseline=baseline,
            stage=stage,
            name=stage,
            split=DatasetSplit.TEST,
        )
        if log_model:
            log_model_with_baseline(model, train, spec.tracking, baseline)
    logger.info("Completed %s stage", stage)
    return _StageResult(model, params, scores)


def _log_summary(
    validation_mean: dict[str, float], test_metrics: dict[str, float] | None
) -> None:
    metrics(
        (
            "Evaluation summary (test scores belong to the main model)"
            if test_metrics is not None
            else "Evaluation summary"
        ),
        (
            {"validation mean": validation_mean, "test": test_metrics}
            if test_metrics is not None
            else {"validation mean": validation_mean}
        ),
        verbosity="normal",
    )
