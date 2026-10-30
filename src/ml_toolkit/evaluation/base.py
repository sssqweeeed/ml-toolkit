"""Common contract for reports consuming aligned predictions."""

from dataclasses import dataclass
from pathlib import Path
import re

import polars as pl

from ..config import FigureConfig
from ..data import Pool
from ..feature_levels import FeatureLevelConfig
from ..types import FloatArray


@dataclass(frozen=True, kw_only=True)
class PredictionEvaluator:
    """Named additional metrics and artifacts; neither method is mandatory.

    ``evaluate`` keys are logged under ``<name>/<key>`` and do not become CV
    averages or optimization objectives. Return finite scalar values only.
    ``write_artifacts`` receives shared figure and feature-level settings.
    Only returned files are uploaded under ``<name>/``. The directory is deleted
    after uploading.
    """

    name: str

    def __post_init__(self) -> None:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.name) is None:
            raise ValueError(
                "Evaluator name must start with an ASCII letter or digit and contain only ASCII letters, digits, underscores, hyphens or dots"
            )

    def evaluate(
        self, *, data: Pool, prediction: FloatArray, split: pl.Series
    ) -> dict[str, float]:
        """Return additional scalar metrics, or an empty dictionary.

        Parameters
        ----------
        data : Pool
            Combined train and optional held-out rows. Do not mutate its fields.
        prediction : FloatArray
            Finite one-dimensional predictions aligned with data, shape (n_rows,).
        split : polars.Series
            Aligned non-null String labels: train plus at most one of val/test.

        Returns
        -------
        dict of str to float
            Finite values logged as ``<name>/<key>``. Include the sample in keys
            when relevant; these values do not become optimization objectives.
        """
        return {}

    def write_artifacts(
        self,
        *,
        data: Pool,
        prediction: FloatArray,
        split: pl.Series,
        artifact_dir: Path,
        figure: FigureConfig,
        features: tuple[FeatureLevelConfig, ...] | None,
        n_bins: int,
    ) -> list[Path]:
        """Write report files and return only the paths to upload.

        Parameters
        ----------
        data : Pool
            Combined train and optional held-out rows; inputs must not be mutated.
        prediction : FloatArray
            Finite predictions aligned with data, shape (n_rows,).
        split : polars.Series
            Aligned non-null String labels: train plus at most one of val/test.
        artifact_dir : pathlib.Path
            Temporary directory owned by the caller, removed after upload.
        figure : FigureConfig
            Shared figure dimensions, resolution and file format.
        features : tuple of FeatureLevelConfig or None
            Shared feature-level rules. None infers rules; an empty tuple disables
            level diagnostics. Use only if relevant to this report.
        n_bins : int
            Shared numerical feature-level bin limit.

        Returns
        -------
        list of pathlib.Path
            Files relative to artifact_dir, or absolute paths inside it. Only
            these files are uploaded under the report name. Return [] to skip.
        """
        return []

    def parameters(self) -> dict[str, object]:
        """Return an explicit JSON-compatible settings description, never state."""
        return {}
