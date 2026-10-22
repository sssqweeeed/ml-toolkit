"""Example data measurements and a table without model predictions."""

from pathlib import Path

import polars as pl

from ml_toolkit.config import FigureConfig
from ml_toolkit.data import Pool
from ml_toolkit.data_analysis import DataAnalyzer
from ml_toolkit.feature_levels import FeatureLevelConfig


class DataAudit(DataAnalyzer):
    """Count sample rows and save per-feature null counts."""

    def evaluate(self, *, data: Pool, split: pl.Series) -> dict[str, float]:
        return {
            f"{part}/rows": float((split == part).sum())
            for part in split.unique(maintain_order=True)
        }

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
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = Path("null_counts.csv")
        data.X.null_count().write_csv(artifact_dir / path)
        return [path]
