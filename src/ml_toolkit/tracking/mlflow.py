"""Small MLflow artifact helpers shared by training diagnostics."""

import logging

import mlflow
import polars as pl

logger = logging.getLogger(__name__)


def log_polars_table(table: pl.DataFrame, artifact_file: str) -> None:
    """Log records without imposing pandas conversion on PSI results."""
    mlflow.log_dict({"rows": table.to_dicts()}, artifact_file)
    logger.debug("Logged table %s: %s rows, %s columns", artifact_file, table.height, table.width)
