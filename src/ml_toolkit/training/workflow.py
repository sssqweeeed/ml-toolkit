"""High-level selection, optimization, validation and final/test orchestration."""

import logging
from collections.abc import Callable
from dataclasses import asdict
from datetime import date

from optuna import Trial

from ..config import HoldoutConfig, TrainingConfig
from ..data import validate_training_data
from ..metrics import Metric
from ..models.base import Model
from ..tracking import runs as tracking
from ..types import ModelParams
from .contracts import (
    DatasetSplit,
    FeatureRanker,
    PostFitEvaluator,
    Pool,
)
from .evaluation import evaluate_model
from .fit import fit_final_model, fit_split
from .results import SplitResult, TrainingResult, ValidationResult
from .splits import make_validation_splits, split_holdout

logger = logging.getLogger(__name__)


def train_model(
    *,
    data: Pool,
    params: ModelParams,
    model_factory: Callable[[ModelParams], Model],
    metric: Metric,
    run_name: str,
    config: TrainingConfig,
    feature_ranker: FeatureRanker | None = None,
    param_space: Callable[[Trial], ModelParams] | None = None,
    post_fit_evaluator: PostFitEvaluator | None = None,
) -> TrainingResult:
    """Tune, validate, refit and optionally test a complete training workflow.

    Parameters
    ----------
    data : Pool
        Aligned prepared features, finite targets, ascending non-null timestamps
        and nonnegative weights with positive total. May include a future test tail.
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
        Holdout/TSCV, test row count, fraction or calendar start date, selection,
        tuning, reporting and packaging. A test start date includes the entire
        day in the timezone of data.time.
    feature_ranker : FeatureRanker or None, default None
        Required for selection on each split's validation rows.
    param_space : Callable or None, default None
        Optuna trial overrides; required when tuning is enabled.
    post_fit_evaluator : PostFitEvaluator or None, default None
        Reused custom evaluator; enabled calls require aligned object IDs.

    Returns
    -------
    TrainingResult
        Final model, split results, last-split settings and optional test scores.

    Notes
    -----
    Each split independently selects features using the base parameters, tunes
    on its validation rows, then fits the winning settings once for reporting.
    Validation averages are reports, not Optuna objectives. Final fitting uses
    the last split's features, parameters, iterations and optional threshold on
    all rows before test, without repeated selection or validation. Test only
    receives predictions after fitting. MLflow URI and experiment are caller-owned.

    Examples
    --------
    >>> from ml_toolkit.config import HoldoutConfig, TrainingConfig
    >>> config = TrainingConfig(validation=HoldoutConfig(), test_size=0.2)
    >>> result = train_model(  # doctest: +SKIP
    ...     data=data, params=params, model_factory=factory, metric=metric,
    ...     run_name="experiment", config=config,
    ... )
    """
    validate_training_data(data)
    train, test = data, None
    test_size = config.test_size
    if isinstance(test_size, date):
        train_rows = int((data.time.dt.date() < test_size).sum())
        if train_rows == 0 or train_rows == len(data):
            raise ValueError("test_size date must leave nonempty training and test data")
        test_size = len(data) - train_rows
    if isinstance(test_size, (int, float)):
        partition = split_holdout(data, HoldoutConfig(val_size=test_size))
        train, test = partition.train, partition.val
    splits = make_validation_splits(train, config.validation)
    if config.feature_selection is not None and feature_ranker is None:
        raise ValueError("feature_ranker is required when feature selection is enabled")
    if config.tuning is not None and param_space is None:
        raise ValueError("param_space is required when tuning is enabled")
    policy = config.reporting.post_fit
    if (
        post_fit_evaluator is not None
        and policy.stages
        and policy.splits
        and data.object_id is None
    ):
        raise ValueError("object_id is required when post_fit_evaluator is supplied")

    logger.info("Starting training run %s", run_name)
    logger.debug(
        "Training data: %s rows, %s features", len(train), len(train.X.columns)
    )
    with tracking.run(run_name):
        tracking.log_json(asdict(config), "configuration/requested.json")
        diagnosed = config.reporting.validation_splits
        if diagnosed == "all":
            diagnosed_indices = {part.index for part in splits}
        elif diagnosed == "last":
            diagnosed_indices = {splits[-1].index}
        else:
            diagnosed_indices = set(diagnosed)
        logger.info("Starting validation with %s evaluated splits", len(splits))
        with tracking.run("validation"):
            split_results: list[SplitResult] = []
            for part in splits:
                split_results.append(
                    fit_split(
                        train=part.train,
                        validation=part.val,
                        params=params,
                        model_factory=model_factory,
                        metric=metric,
                        config=config,
                        feature_ranker=feature_ranker,
                        post_fit_evaluator=post_fit_evaluator,
                        param_space=param_space,
                        index=part.index,
                        diagnostics=part.index in diagnosed_indices,
                    )
                )
            validation = ValidationResult(tuple(split_results))
            mean_metrics = validation.mean_metrics
            tracking.log_metrics(
                {f"validation/{key}": value for key, value in mean_metrics.items()}
            )
        logger.info("Completed validation with %s splits", len(splits))
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
        with tracking.run("final_fit"):
            model, final_train, final_params = fit_final_model(
                data=train,
                last_split=last,
                model_factory=model_factory,
            )
            final_test = None if test is None else test.select_features(last.features)
            scores = evaluate_model(
                model,
                final_train,
                final_test,
                metric,
                config.reporting,
                post_fit_evaluator,
                stage="final",
                name="final_fit",
                split=DatasetSplit.TEST,
                diagnostics=test is not None,
            )
            if config.tracking.log_final_model:
                model.log(final_train, config=config.tracking)
            logger.info("Completed final fit for run %s", run_name)
        logger.info("Completed training run %s", run_name)
        return TrainingResult(
            model=model,
            validation=validation,
            params=final_params,
            features=last.features,
            feature_selection=last.feature_selection,
            tuning=last.tuning,
            test_metrics=scores.heldout_metrics,
        )
