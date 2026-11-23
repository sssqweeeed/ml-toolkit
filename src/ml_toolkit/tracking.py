"""MLflow helpers: table artifacts and PSI baseline loading."""

from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

import mlflow
import mlflow.artifacts
import polars as pl
from mlflow.models import Model

from ._console import Logger
from .psi import PSIBaseline

logger = Logger(__name__)


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


def load_psi_baseline(model_uri: str) -> PSIBaseline:
    """Load the named PSI artifact from a model URI or local model directory.

    Accepts MLflow model IDs, registered versions and run model URIs. Prefer
    immutable versions when loading predictions and monitoring separately.
    Does not import or instantiate the predictive model. Missing baselines
    raise FileNotFoundError; malformed or unsupported state raises ValueError.
    """
    source = model_uri.rstrip("/")
    if source.startswith("models:/"):
        identifier = source.removeprefix("models:/")
        if "/" not in identifier and "@" not in identifier:
            # A model ID URI has no version segment; appending /MLmodel would
            # make MLflow interpret the ID as a registered model name instead.
            source = mlflow.get_logged_model(identifier).artifact_location.rstrip("/")
    with TemporaryDirectory(prefix="ml-toolkit-psi-load-") as directory:
        # Reading MLmodel only avoids downloading weights or executing entrypoints.
        descriptor = mlflow.artifacts.download_artifacts(
            artifact_uri=f"{source}/MLmodel", dst_path=directory
        )
        model = Model.load(descriptor)
        flavors = model.flavors
        artifact = (
            flavors.get("python_function", {}).get("artifacts", {}).get("psi_baseline")
        )
        if artifact is None:
            raise FileNotFoundError(
                f"Model {model_uri!r} has no PSI baseline: it may predate this feature, "
                "have reporting.psi=False, or have levels disabled with features=()."
            )
        relative = artifact.get("path")
        if not isinstance(relative, str) or not relative or "\\" in relative:
            raise ValueError("Invalid PSI artifact path in MLmodel")
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or ":" in relative:
            raise ValueError("PSI artifact path must be relative to the model package")
        baseline_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"{source}/{path.as_posix()}",
            dst_path=str(Path(directory) / "baseline"),
        )
        return PSIBaseline.load(baseline_path)
