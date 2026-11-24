"""MLflow helpers: runs, JSON and table artifacts."""

import json

import mlflow
import polars as pl
from mlflow.tracking.fluent import ActiveRun

from ._console import Logger

logger = Logger(__name__)


def nested_run(name: str) -> ActiveRun:
    """Start an MLflow run, nested inside the active run when there is one.

    Parameters
    ----------
    name : str
        Run name.

    Returns
    -------
    mlflow.tracking.fluent.ActiveRun
        The started run, usable as a context manager.
    """
    return mlflow.start_run(run_name=name, nested=mlflow.active_run() is not None)


def log_json(payload: object, artifact_file: str) -> None:
    """Log a JSON-serializable payload as an artifact of the active run.

    Parameters
    ----------
    payload : object
        Value written with ``json.dumps``; values JSON cannot represent, such
        as paths or dates, are stored as strings.
    artifact_file : str
        Artifact-relative file path in the MLflow run.
    """
    mlflow.log_text(json.dumps(payload, default=str, indent=2), artifact_file)


def log_polars_table(table: pl.DataFrame, artifact_file: str) -> None:
    """Log Polars records as a JSON artifact without converting to pandas.

    Parameters
    ----------
    table : polars.DataFrame
        Table whose rows can be serialized by MLflow.
    artifact_file : str
        Artifact-relative JSON file path in the MLflow run.

    Notes
    -----
    Calls mlflow.log_dict with a {"rows": table.to_dicts()} payload. Uses the
    active run, or MLflow's implicit run if no run is active.
    """
    mlflow.log_dict({"rows": table.to_dicts()}, artifact_file)
    logger.debug(
        "Logged table %s: %s rows, %s columns", artifact_file, table.height, table.width
    )
