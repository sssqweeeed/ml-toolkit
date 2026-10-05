"""Small MLflow artifact helpers shared by training diagnostics."""

import logging

import mlflow
import polars as pl

logger = logging.getLogger(__name__)


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
    logger.debug("Logged table %s: %s rows, %s columns", artifact_file, table.height, table.width)
