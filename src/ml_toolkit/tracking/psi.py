"""Load a bundled PSI reference without executing model serving code."""

from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

import mlflow.artifacts
from mlflow.models import Model

from ..psi import PSIBaseline


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
