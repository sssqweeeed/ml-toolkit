"""Common contract for reports that require data without predictions."""

from dataclasses import dataclass
from pathlib import Path
import re
import polars as pl
from ..config import FigureConfig
from ..data import Pool
from ..feature_levels import FeatureLevelConfig


@dataclass(frozen=True, kw_only=True)
class DataAnalyzer:
    """Named data metrics and artifacts, independent of model predictions.

    Implement either hook or both. During training, finite metrics are logged
    under ``<name>/<key>`` and only returned files are uploaded under ``<name>/``.
    These metrics do not become CV averages or optimization objectives.
    Names must be unique across analyzers and prediction evaluators.
    """

    name: str

    def __post_init__(self) -> None:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.name) is None:
            raise ValueError(
                "Analyzer name must start with an ASCII letter or digit and contain only ASCII letters, digits, underscores, hyphens or dots"
            )

    def evaluate(self, *, data: Pool, split: pl.Series) -> dict[str, float]:
        """Return finite scalar measurements, or an empty dictionary.

        Parameters
        ----------
        data : Pool
            Combined train and optional held-out rows. Do not mutate its fields.
        split : polars.Series
            Aligned non-null String labels: train plus at most one of val/test.

        Returns
        -------
        dict of str to float
            Additional values logged as ``<name>/<key>``. Include sample names
            in keys where relevant.
        """
        return {}

    def write_artifacts(
        self,
        *,
        data: Pool,
        split: pl.Series,
        artifact_dir: Path,
        figure: FigureConfig,
        features: tuple[FeatureLevelConfig, ...] | None,
        n_bins: int,
    ) -> list[Path]:
        """Write data reports and return only files to upload.

        Parameters
        ----------
        data : Pool
            Combined train and optional held-out rows; inputs must not be mutated.
        split : polars.Series
            Aligned non-null String labels: train plus at most one of val/test.
        artifact_dir : pathlib.Path
            Temporary directory owned by the caller, removed after uploading.
        figure : FigureConfig
            Shared figure dimensions, resolution and format.
        features : tuple of FeatureLevelConfig or None
            Shared feature-level rules; None infers rules and () disables levels.
            Use only if relevant to this analysis.
        n_bins : int
            Shared numerical feature-level bin limit.

        Returns
        -------
        list of pathlib.Path
            Files relative to artifact_dir, or absolute paths inside it.
            Only these files are uploaded under the analyzer name.
        """
        return []

    def parameters(self) -> dict[str, object]:
        """Return explicit JSON-compatible settings, never data or fitted state."""
        return {}
