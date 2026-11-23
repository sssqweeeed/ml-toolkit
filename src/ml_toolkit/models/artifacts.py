"""MLflow models-from-code packaging and its settings."""

import importlib
import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol, cast

import catboost
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.models.model import ModelInfo

from .._console import Logger
from ..types import FloatArray, ModelParams
from .state import ModelState

PACKAGES = (
    "catboost",
    "polars",
    "pyarrow",
    "matplotlib",
    "scikit-learn",
)

logger = Logger(__name__)


@dataclass(frozen=True, kw_only=True, slots=True)
class TrackingConfig:
    """Configure model packaging in an externally configured MLflow store.

    Parameters
    ----------
    log_trial_models : bool, default False
        Package each trial model in each split's Optuna study.
    log_validation_models : bool, default False
        Package each split's post-tuning validation model.
    log_test_model : bool, default True
        Package the model fitted without test and evaluated on held-out test rows.
    log_final_model : bool, default True
        Package a model fitted on all eligible rows, including when no test exists.
    input_example_rows : int, default 5
        Positive maximum serving-example row count, capped at sample size.
    model_name : str, default 'model'
        Nonempty MLflow model artifact name shared across nested runs.
    registered_model_name : str or None, default None
        Registry name for the additional model when fit_on_test is enabled,
        otherwise for the main model. Requires that model's logging flag:
        log_final_model with fit_on_test; otherwise log_test_model when a test
        sample exists, or log_final_model when it does not. TrainingConfig and
        train_model raise ValueError instead of silently skipping registration.
    conda_env : pathlib.Path or None, default None
        Existing serving environment file; None pins installed dependencies.

    Notes
    -----
    Metrics, run metadata, and configuration artifacts are always logged.
    The caller owns tracking URI, experiment selection, and Python logging.
    """

    log_trial_models: bool = False
    log_validation_models: bool = False
    log_test_model: bool = True
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


def log_model_from_code(
    *,
    model_class: type,
    model_module: str = "ml_toolkit.models",
    sample_frame: pd.DataFrame,
    sample_prediction: FloatArray,
    artifacts: dict[str, str],
    metadata: dict[str, object] | None,
    params: ModelParams,
    name: str,
    registered_model_name: str | None = None,
    conda_env: Path | None = None,
    code_paths: list[str] | None = None,
    packages: tuple[str, ...] = (),
    extra_artifacts: Mapping[str, str] | None = None,
) -> ModelInfo:
    if (
        getattr(importlib.import_module(model_module), model_class.__name__, None)
        is not model_class
    ):
        raise ValueError(
            f"{model_class.__name__} is not exported by {model_module}; only "
            "toolkit adapters exported from that module can be logged this way"
        )
    artifacts = merge_artifacts(artifacts, extra_artifacts)
    if sample_frame.empty:
        raise ValueError("The logging sample must contain at least one row.")

    environment_path: str | None = None
    requirements: list[str] | None = None
    if conda_env is not None:
        conda_env = Path(conda_env).resolve()
        if not conda_env.is_file():
            raise FileNotFoundError(f"Environment file not found: {conda_env}")
        environment_path = str(conda_env)
    else:
        requirements = [
            f"{package}=={version(package)}"
            for package in dict.fromkeys(
                ("mlflow", "numpy", "pandas", *PACKAGES, *packages)
            )
        ]

    logger.debug(
        "Logging model artifact %s: %s sample rows, registered=%s",
        name,
        len(sample_frame),
        registered_model_name is not None,
    )
    with TemporaryDirectory(prefix="ml-toolkit-package-") as directory:
        # Models-from-code loads this file; it only exposes the adapter class
        # that already lives in the logged ml_toolkit package.
        class_name = model_class.__name__
        entrypoint = Path(directory) / f"{class_name}.py"
        entrypoint.write_text(
            "from mlflow.models import set_model\n\n"
            f"from {model_module} import {class_name}\n\n"
            f"set_model({class_name}())\n",
            encoding="utf-8",
        )
        package = Path(directory) / "ml_toolkit"
        shutil.copytree(
            Path(__file__).resolve().parents[1],
            package,
            ignore=shutil.ignore_patterns(
                "__pycache__",
                "*.py[co]",
            ),
        )
        model_info = mlflow.pyfunc.log_model(
            python_model=str(entrypoint),
            name=name,
            registered_model_name=registered_model_name,
            signature=infer_signature(sample_frame, sample_prediction),
            input_example=sample_frame,
            params=params,
            artifacts=artifacts,
            metadata=metadata,
            code_paths=[str(package), *(code_paths or [])],
            conda_env=environment_path,
            pip_requirements=requirements,
        )
    logger.debug("Logged model artifact %s", name)
    return model_info


def merge_artifacts(
    artifacts: Mapping[str, str], extra_artifacts: Mapping[str, str] | None
) -> dict[str, str]:
    """Merge extra artifacts into a model's artifacts without overwriting any.

    Parameters
    ----------
    artifacts : Mapping[str, str]
        Artifact names and local paths written by the model adapter.
    extra_artifacts : Mapping[str, str] or None
        Additional named artifacts, for example a PSI baseline.

    Returns
    -------
    dict[str, str]
        New dictionary containing both sets of artifacts.

    Raises
    ------
    ValueError
        If an extra artifact reuses a model artifact name.
    """
    extra: Mapping[str, str] = {} if extra_artifacts is None else extra_artifacts
    overlap = artifacts.keys() & extra.keys()
    if overlap:
        raise ValueError(
            f"Extra artifacts conflict with model artifacts: {sorted(overlap)}"
        )
    return {**artifacts, **extra}


class NativeModelState(Protocol):
    @property
    def native_model(self) -> catboost.CatBoost: ...
    def export_state(self) -> ModelState: ...


def log_catboost(
    model: NativeModelState,
    sample_frame: pd.DataFrame,
    sample_prediction: FloatArray,
    *,
    name: str,
    registered_model_name: str | None = None,
    conda_env: Path | None = None,
    metadata: dict[str, object] | None = None,
    extra_artifacts: Mapping[str, str] | None = None,
) -> ModelInfo:
    state = model.export_state()
    with TemporaryDirectory(prefix="ml-toolkit-model-") as directory:
        directory = Path(directory)
        model_path = directory / "model.cbm"
        model.native_model.save_model(str(model_path))
        config_path = directory / "config.json"
        serialized_state = json.dumps(state, default=str)
        config_path.write_text(serialized_state, encoding="utf-8")
        logged_state = cast(dict[str, object], json.loads(serialized_state))
        return log_model_from_code(
            model_class=type(model),
            sample_frame=sample_frame,
            sample_prediction=sample_prediction,
            artifacts={"model": str(model_path), "config": str(config_path)},
            extra_artifacts=extra_artifacts,
            metadata={**logged_state, **(metadata or {})},
            params=model.native_model.get_all_params(),
            name=name,
            registered_model_name=registered_model_name,
            conda_env=conda_env,
        )
