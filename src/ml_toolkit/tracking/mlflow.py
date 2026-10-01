"""Small MLflow artifact helpers shared by training diagnostics."""

import mlflow
import polars as pl


def log_polars_table(table: pl.DataFrame, artifact_file: str) -> None:
    """Log records without imposing pandas conversion on PSI results."""
    mlflow.log_dict({"rows": table.to_dicts()}, artifact_file)
