"""Selection, optimization and high-level workflow settings."""

from dataclasses import dataclass, field
from math import isfinite
from typing import TYPE_CHECKING, Literal

from .._console import Verbosity
from ..models import TrackingConfig
from .reporting import ReportingConfig

if TYPE_CHECKING:
    from ..reports import PredictionEvaluator


@dataclass(frozen=True, kw_only=True, slots=True)
class TuningConfig:
    """Configure each split's independent Optuna study.

    Parameters
    ----------
    metric : str
        Validation metric key, or the exact key returned by evaluator.evaluate()
        when evaluator is supplied. Do not prepend the evaluator's name.
    direction : {'minimize', 'maximize'}
        Direction of the objective.
    n_trials : int, default 50
        Positive maximum number of trials per split.
    timeout_seconds : float or None, default None
        Positive time limit per split; a running trial may exceed it.
    seed : int or None, default None
        Nonnegative TPE sampler seed, independent of model seeds.
    evaluator : PredictionEvaluator or None, default None
        Optional objective source for train_model. Called on combined train and
        validation rows for every trial, independently of reporting flags. Select
        validation rows using split labels. To enable reports, also pass this same
        instance through prediction_evaluators. None uses the ordinary metric.
    """

    metric: str
    direction: Literal["minimize", "maximize"]
    n_trials: int = 50
    timeout_seconds: float | None = None
    seed: int | None = None
    evaluator: "PredictionEvaluator | None" = None

    def __post_init__(self) -> None:
        if self.n_trials < 1:
            raise ValueError("n_trials must be positive")
        if not self.metric.strip():
            raise ValueError("metric must not be empty")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if self.timeout_seconds is not None and (
            not isfinite(self.timeout_seconds) or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive")


def check_registration_target_logged(
    tracking: TrackingConfig, *, fit_on_test: bool, has_test_sample: bool | None
) -> None:
    """Raise when registered_model_name targets a model that is not logged.

    has_test_sample None checks only the cases that fail for any test sample.
    """
    if tracking.registered_model_name is None:
        return
    if fit_on_test:
        if not tracking.log_final_model:
            raise ValueError(
                "registered_model_name targets the additional final model when "
                "fit_on_test=True; enable log_final_model"
            )
    elif has_test_sample is None:
        if not (tracking.log_test_model or tracking.log_final_model):
            raise ValueError(
                "registered_model_name targets the main model, which is never "
                "logged; enable log_test_model or log_final_model"
            )
    elif has_test_sample and not tracking.log_test_model:
        raise ValueError(
            "registered_model_name targets the main model, which is not logged "
            "when a test sample exists; enable log_test_model"
        )
    elif not has_test_sample and not tracking.log_final_model:
        raise ValueError(
            "registered_model_name targets the main model, which is not logged "
            "without a test sample; enable log_final_model"
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class TrainingConfig:
    """Configure parameter tuning, additional reports and model packaging.

    Parameters
    ----------
    tuning : TuningConfig or None, default None
        Independent search per validation split, requiring param_space.
    reporting : ReportingConfig
        Additional metric and artifact settings.
    tracking : TrackingConfig
        Model packaging settings.
    fit_on_test : bool, default False
        After independent test evaluation, fit an additional fresh model on
        all eligible data including test. Requires a held-out test sample.
    verbosity : {'silent', 'normal', 'verbose', 'debug'}, default 'normal'
        Console output for this training run: none, milestones and summary,
        split/trial details, or internal diagnostics. Writes to stderr without
        configuring Python logging or changing CatBoost, Optuna or MLflow output.
        Warnings, exceptions and application logging remain independent.

    Notes
    -----
    Both post-validation fits use the last split's features and refit parameters.
    """

    tuning: TuningConfig | None = None
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    fit_on_test: bool = False
    verbosity: Verbosity = "normal"

    def __post_init__(self) -> None:
        if self.verbosity not in ("silent", "normal", "verbose", "debug"):
            raise ValueError("verbosity must be one of: silent, normal, verbose, debug")
        check_registration_target_logged(
            self.tracking, fit_on_test=self.fit_on_test, has_test_sample=None
        )
