"""Common contract for fitted-model reports."""

from dataclasses import dataclass
from pathlib import Path
import re

import polars as pl

from ..config import FigureConfig
from ..data import Pool
from ..metrics import Metric
from ..models.base import Model, ShrinkableModel


@dataclass(frozen=True, kw_only=True)
class ModelEvaluator:
    """Named model report with optional scalar metrics and selected artifacts.

    Both methods receive the fitted adapter, the training metric callback and
    aligned data/split labels. Do not mutate inputs or retain fitted state:
    evaluator instances are reused across folds. Scalar results are additional
    reports, never training objectives or validation averages. Only returned
    artifact paths inside artifact_dir are uploaded beneath name/.
    """

    name: str

    def __post_init__(self) -> None:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.name) is None:
            raise ValueError(
                "Evaluator name must start with an ASCII letter or digit and contain only ASCII letters, digits, underscores, hyphens or dots"
            )

    def evaluate(
        self,
        *,
        model: Model | ShrinkableModel,
        metric: Metric,
        data: Pool,
        split: pl.Series,
    ) -> dict[str, float]:
        """Return finite additional scalar scores, or {} to omit them."""
        return {}

    def write_artifacts(
        self,
        *,
        model: Model | ShrinkableModel,
        metric: Metric,
        data: Pool,
        split: pl.Series,
        artifact_dir: Path,
        figure: FigureConfig,
    ) -> list[Path]:
        """Write report files and return relative or contained absolute paths."""
        return []

    def parameters(self) -> dict[str, object]:
        """Return JSON-compatible settings, excluding fitted state."""
        return {}
