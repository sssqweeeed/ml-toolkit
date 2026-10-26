"""High-level selection, optimization, validation and final/test orchestration."""

import logging
from collections.abc import Callable
from dataclasses import asdict, replace

from optuna import Trial

from ..config import TrainingConfig
from ..data import validate_training_data
from ..feature_selection import FeatureSelector
from ..metrics import Metric
from ..models.base import Model
from ..tracking import runs as tracking
from ..types import ModelParams
from .contracts import (
    DatasetSplit,
    Pool,
)
from .evaluation import evaluate_model
from .psi import prepare_psi_baseline, log_model_with_baseline
from .reporting import prepare_evaluators, prepare_analyzers
from ..evaluation import PredictionEvaluator
from ..data_analysis import DataAnalyzer
from .fit import refit_model, fit_split
from .results import SplitResult, TrainingResult, ValidationResult
from .splits import Splitter

logger = logging.getLogger(__name__)


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
        Parameter tuning, additional reporting and model packaging settings.
    feature_selector : FeatureSelector or None, default None
        Selects features independently before tuning on each split. None keeps
        all columns. The selector owns its settings and must not retain fitted state.
    param_space : Callable or None, default None
        Optuna trial overrides; required when tuning is enabled.
    prediction_evaluators : PredictionEvaluator, list[PredictionEvaluator], or None, default None
        Named evaluators called in list order with one combined pool and aligned
        train and optional val or test labels per fit. Names are unique ignoring
        case; object IDs are optional. ReportingConfig stage flags gate all reports.
        None or an empty list disables evaluator calls, not data analysis or PSI/SHAP.
    data_analyzers : DataAnalyzer, list[DataAnalyzer], or None, default None
        Data reports receiving the same pool and split labels without predictions.
        Called in list order after prediction evaluators, using the same stage
        flags and selected features. Names must be unique across both lists and
        enabled PSI/SHAP. None or [] disables custom data reports, not PSI.

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
    splits = splitter.get_validation()
    part = next(splits, None)
    if part is None:
        raise ValueError("Splitter must yield at least one validation split")
    if config.tuning is not None and param_space is None:
        raise ValueError("param_space is required when tuning is enabled")
    analyzers = prepare_analyzers(data_analyzers)
    evaluators = prepare_evaluators(prediction_evaluators, config.reporting, analyzers)

    logger.info("Starting training run %s", run_name)
    with tracking.run(run_name):
        tracking.log_json(asdict(config), "configuration/requested.json")
        tracking.log_json(
            {
                "class": f"{type(splitter).__module__}.{type(splitter).__qualname__}",
                "parameters": splitter.parameters(),
            },
            "configuration/splitter.json",
        )
        tracking.log_json(
            [
                {
                    "name": item.name,
                    "class": f"{type(item).__module__}.{type(item).__qualname__}",
                    "parameters": item.parameters(),
                }
                for item in evaluators
            ],
            "configuration/evaluators.json",
        )
        tracking.log_json(
            [
                {
                    "name": item.name,
                    "class": f"{type(item).__module__}.{type(item).__qualname__}",
                    "parameters": item.parameters(),
                }
                for item in analyzers
            ],
            "configuration/data_analyzers.json",
        )
        logger.info("Starting validation")
        with tracking.run("validation"):
            split_results: list[SplitResult] = []
            indices: set[int] = set()
            while part is not None:
                if part.heldout is None:
                    raise ValueError("Validation split must contain heldout data")
                if part.index is None or part.index <= 0 or part.index in indices:
                    raise ValueError(
                        "Validation split index must be positive and unique"
                    )
                indices.add(part.index)
                split_results.append(
                    fit_split(
                        train=part.train,
                        validation=part.heldout,
                        params=params,
                        model_factory=model_factory,
                        metric=metric,
                        config=config,
                        feature_selector=feature_selector,
                        evaluators=evaluators,
                        analyzers=analyzers,
                        param_space=param_space,
                        index=part.index,
                    )
                )
                del part
                part = next(splits, None)
            validation = ValidationResult(tuple(split_results))
            mean_metrics = validation.mean_metrics
            tracking.log_metrics(
                {f"validation/{key}": value for key, value in mean_metrics.items()}
            )
        logger.info("Completed validation with %s splits", len(split_results))
        logger.debug("Validation mean metrics: %s", mean_metrics)
        last = validation.last_split
        tracking.log_json(
            {
                "features": last.features,
                "model_params": last.refit_params,
                "validation_split": last.index,
            },
            "configuration/resolved.json",
        )
        test = splitter.get_test()
        if config.fit_on_test and test.heldout is None:
            raise ValueError("fit_on_test=True requires a held-out test sample")
        test_train = test.train.select_features(last.features)
        test_heldout = (
            None
            if test.heldout is None
            else test.heldout.select_features(last.features)
        )
        del test
        stage = "final_fit" if test_heldout is None else "test"
        logger.info("Starting %s stage", stage)
        with tracking.run(stage):
            model, final_params = refit_model(
                data=test_train,
                last_split=last,
                model_factory=model_factory,
            )
            log_main = (
                config.tracking.log_final_model
                if test_heldout is None
                else config.tracking.log_test_model
            )
            baseline = prepare_psi_baseline(
                test_train, config.reporting, log_model=log_main, stage=stage
            )
            scores = evaluate_model(
                model,
                test_train,
                test_heldout,
                metric,
                config.reporting,
                evaluators,
                analyzers=analyzers,
                psi_baseline=baseline,
                stage=stage,
                name=stage,
                split=DatasetSplit.TEST,
            )
            if log_main:
                main_tracking = (
                    replace(config.tracking, registered_model_name=None)
                    if config.fit_on_test
                    else config.tracking
                )
                log_model_with_baseline(model, test_train, main_tracking, baseline)
        logger.info("Completed %s stage", stage)
        test_schema = (
            test_train.X.schema,
            test_train.cat_features,
            test_train.numerical_features,
        )
        del test_train, test_heldout
        final_model = None
        if config.fit_on_test:
            full = splitter.get_final()
            validate_training_data(full)
            full = full.select_features(last.features)
            if (
                full.X.schema,
                full.cat_features,
                full.numerical_features,
            ) != test_schema:
                raise ValueError(
                    "Final-fit data must match the selected training schema"
                )
            logger.info("Starting final_fit stage")
            with tracking.run("final_fit"):
                final_model, _ = refit_model(
                    data=full, last_split=last, model_factory=model_factory
                )
                baseline = prepare_psi_baseline(
                    full,
                    config.reporting,
                    log_model=config.tracking.log_final_model,
                    stage="final_fit",
                )
                evaluate_model(
                    final_model,
                    full,
                    None,
                    metric,
                    config.reporting,
                    evaluators,
                    analyzers=analyzers,
                    psi_baseline=baseline,
                    stage="final_fit",
                    name="final_fit",
                )
                if config.tracking.log_final_model:
                    log_model_with_baseline(
                        final_model, full, config.tracking, baseline
                    )
            logger.info("Completed final_fit stage")
        logger.info("Completed training run %s", run_name)
        return TrainingResult(
            model=model,
            validation=validation,
            params=final_params,
            features=last.features,
            feature_selection=last.feature_selection,
            tuning=last.tuning,
            test_metrics=scores.heldout_metrics,
            final_model=final_model,
        )
