"""Common MLflow models-from-code packaging for native model artifacts."""

from collections.abc import Mapping
import json
import shutil
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


def log_model_from_code(
    *,
    source: str | Path,
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
        package = Path(directory) / "ml_toolkit"
        shutil.copytree(
            Path(__file__).resolve().parents[1],
            package,
            ignore=shutil.ignore_patterns(
                "__pycache__",
                "*.py[co]",
                ".pytest_cache",
                ".mypy_cache",
                ".ruff_cache",
                ".cache",
                ".codex",
                ".codegraph",
                ".agents",
                "AGENTS.md",
            ),
        )
        model_info = mlflow.pyfunc.log_model(
            python_model=str(source),
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
    """Merge explicitly supplied artifacts without overwriting model state."""
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
        source = (
            Path(__file__).resolve().parent
            / "entrypoints"
            / f"{type(model).__name__}.py"
        )
        return log_model_from_code(
            source=source,
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
