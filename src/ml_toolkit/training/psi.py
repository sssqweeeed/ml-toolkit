"""Prepare and package each saved model's PSI reference from its own train pool."""

from pathlib import Path
from tempfile import TemporaryDirectory

from ..config import ReportingConfig, TrackingConfig
from ..feature_levels.config import feature_definitions
from ..models.base import Model
from ..psi import PSIBaseline, PSIConfig
from .contracts import Pool


def prepare_psi_baseline(
    train: Pool, reporting: ReportingConfig, *, log_model: bool, stage: str
) -> PSIBaseline | None:
    if not log_model or not reporting.psi:
        return None
    features = feature_definitions(train, reporting.features)
    if not features:
        return None
    return PSIBaseline.from_reference(
        train.to_analysis_frame(),
        config=PSIConfig(features, n_bins=reporting.n_bins, epsilon=reporting.epsilon),
        time=train.time,
        stage=stage,
    )


def log_model_with_baseline(
    model: Model, sample: Pool, config: TrackingConfig, baseline: PSIBaseline | None
) -> None:
    if baseline is None:
        model.log(sample, config=config)
        return
    with TemporaryDirectory(prefix="ml-toolkit-psi-package-") as directory:
        root = Path(directory) / "psi_baseline"
        baseline.save(root)
        model.log(sample, config=config, extra_artifacts={"psi_baseline": str(root)})
