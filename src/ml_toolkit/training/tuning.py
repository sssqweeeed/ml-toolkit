"""Optuna optimization over an injected scalar scoring callback."""

from collections.abc import Callable, Mapping

import numpy as np
import optuna

from .._console import Logger
from ..config import TuningConfig
from ..tracking import runs as tracking
from ..types import ModelParams
from .results import TuningResult

logger = Logger(__name__)


def tune_optuna(
    *,
    base_params: ModelParams,
    param_space: Callable[[optuna.Trial], Mapping[str, object]],
    score: Callable[[ModelParams], float],
    config: TuningConfig,
) -> TuningResult:
    """Optimize parameters without knowing the data or validation strategy.

    Parameters
    ----------
    base_params : ModelParams
        Fixed constructor parameters. Inputs are never mutated.
    param_space : Callable[[optuna.Trial], Mapping[str, object]]
        Trial overrides, including fixed values and Optuna suggestions.
    score : Callable[[ModelParams], float]
        Receives complete copied parameters and returns one finite objective.
    config : TuningConfig
        Direction, trial/time budget and TPE seed. metric is interpreted by
        the workflow callback rather than this optimizer.

    Returns
    -------
    TuningResult
        Complete winning parameters, objective and zero-based trial number.

    Notes
    -----
    Creates a tuning run with nested trial runs. Configure MLflow externally.
    """
    study = optuna.create_study(
        direction=config.direction, sampler=optuna.samplers.TPESampler(seed=config.seed)
    )
    parameters: dict[int, ModelParams] = {}

    def objective(trial: optuna.Trial) -> float:
        logger.debug("Starting tuning trial %s", trial.number + 1)
        overrides = param_space(trial)
        params = {**base_params, **overrides}
        parameters[trial.number] = params.copy()
        with tracking.run(f"trial_{trial.number + 1}"):
            tracking.log_params(params)
            value = score(params.copy())
            if not np.isfinite(value):
                raise ValueError("score must return a finite numeric scalar")
            value = float(value)
            tracking.log_metrics({"objective": value})
            logger.debug(
                "Completed tuning trial %s: objective=%.6g", trial.number + 1, value,
                verbosity="verbose",
            )
            return value

    logger.info(
        "Starting Optuna tuning: %s trials, metric %s, direction %s",
        config.n_trials,
        config.metric,
        config.direction,
    )
    with tracking.run("tuning"):
        study.optimize(
            objective, n_trials=config.n_trials, timeout=config.timeout_seconds
        )
    logger.info(
        "Completed Optuna tuning: best objective=%.6g at trial %s",
        study.best_value,
        study.best_trial.number + 1,
    )
    return TuningResult(
        parameters[study.best_trial.number],
        float(study.best_value),
        study.best_trial.number,
    )
