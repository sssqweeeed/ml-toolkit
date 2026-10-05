"""Common MLflow models-from-code packaging for native model artifacts."""

from importlib.metadata import version
import logging
from pathlib import Path

import mlflow
from mlflow.models import infer_signature

logger = logging.getLogger(__name__)


def log_model_from_code(
    *,
    source,
    sample_frame,
    sample_prediction,
    artifacts,
    metadata,
    params,
    name,
    registered_model_name=None,
    conda_env=None,
    code_paths=None,
    packages=(),
):
    if sample_frame.empty:
        raise ValueError("The logging sample must contain at least one row.")

    if conda_env is not None:
        conda_env = Path(conda_env).resolve()
        if not conda_env.is_file():
            raise FileNotFoundError(f"Environment file not found: {conda_env}")
        environment = {"conda_env": str(conda_env)}
    else:
        environment = {
            "pip_requirements": [
                f"{package}=={version(package)}"
                for package in dict.fromkeys(("mlflow", "numpy", "pandas", *packages))
            ]
        }

    logger.debug(
        "Logging model artifact %s: %s sample rows, registered=%s",
        name,
        len(sample_frame),
        registered_model_name is not None,
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
        code_paths=code_paths,
        **environment,
    )
    logger.debug("Logged model artifact %s", name)
    return model_info
