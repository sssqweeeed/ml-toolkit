"""MLflow model packaging settings."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, kw_only=True, slots=True)
class TrackingConfig:
    """Configure model packaging in an externally configured MLflow store.

    Parameters
    ----------
    log_trial_models : bool, default False
        Package each trial model in each split's Optuna study.
    log_validation_models : bool, default False
        Package each split's post-tuning validation model.
    log_final_model : bool, default True
        Package the model fitted on all retained rows.
    input_example_rows : int, default 5
        Positive maximum serving-example row count, capped at sample size.
    model_name : str, default 'model'
        Nonempty MLflow model artifact name shared across nested runs.
    registered_model_name : str or None, default None
        Registry name used only for final fitting, never trial or validation models.
    conda_env : pathlib.Path or None, default None
        Existing serving environment file; None pins installed dependencies.

    Notes
    -----
    Metrics, run metadata, and configuration artifacts are always logged.
    The caller owns tracking URI, experiment selection, and Python logging.
    """

    log_trial_models: bool = False
    log_validation_models: bool = False
    log_final_model: bool = True
    input_example_rows: int = 5
    model_name: str = "model"
    registered_model_name: str | None = None
    conda_env: Path | None = None

    def __post_init__(self) -> None:
        if self.input_example_rows < 1:
            raise ValueError("input_example_rows must be positive")
        if not self.model_name.strip():
            raise ValueError("model_name must not be empty")
        if (
            self.registered_model_name is not None
            and not self.registered_model_name.strip()
        ):
            raise ValueError("registered_model_name must not be empty")
