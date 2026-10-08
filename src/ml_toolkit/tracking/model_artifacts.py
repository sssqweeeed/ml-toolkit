"""Common MLflow models-from-code packaging for native model artifacts."""

import logging
import shutil
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.models.model import ModelInfo

from ..types import FloatArray, ModelParams

PACKAGES = (
    "catboost",
    "numba",
    "polars",
    "pyarrow",
    "matplotlib",
    "scikit-learn",
    "optuna",
)

logger = logging.getLogger(__name__)


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
) -> ModelInfo:
    """Package native artifacts and serving source as an MLflow Python model.

    Parameters
    ----------
    source : str or pathlib.Path
        Python serving source path; the source must register a PythonModel with
        MLflow.
    sample_frame : pandas.DataFrame
        Nonempty serving input example used for signature inference.
    sample_prediction : array-like
        Example predictions aligned with sample_frame.
    artifacts : dict[str, str]
        Artifact names mapped to local paths for MLflow packaging.
    metadata : dict
        Model metadata passed to MLflow.
    params : dict
        Model parameters passed to MLflow.
    name : str
        MLflow model artifact name.
    registered_model_name : str or None, default None
        Optional registry name.
    conda_env : str, pathlib.Path, or None, default None
        Existing Conda environment file. None generates pinned pip
        requirements.
    code_paths : list[str] or None, default None
        Additional source paths, bundled alongside the full toolkit package.
    packages : tuple[str, ...], default ()
        Additional installed packages to pin alongside the toolkit dependencies
        when conda_env is omitted.

    Returns
    -------
    mlflow.models.ModelInfo
        Model information returned by mlflow.pyfunc.log_model.

    Raises
    ------
    ValueError
        The serving sample is empty.
    FileNotFoundError
        The specified environment file does not exist.

    Notes
    -----
    Infers the signature from the supplied input/output sample and uploads
    source and artifacts to configured MLflow tracking.
    """
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
