"""Additional held-out metrics and a prediction table for a training run."""

from pathlib import Path

import numpy as np
import polars as pl

from ml_toolkit.config import FigureConfig
from ml_toolkit.data import Pool
from ml_toolkit.evaluation import PredictionEvaluator
from ml_toolkit.feature_levels import FeatureLevelConfig
from ml_toolkit.types import FloatArray


class PredictionAudit(PredictionEvaluator):
    """Log weighted held-out MAE and save the rows behind that score."""

    def evaluate(
        self, *, data: Pool, prediction: FloatArray, split: pl.Series
    ) -> dict[str, float]:
        mask = (split != "train").to_numpy()
        if not mask.any():
            return {}
        part = split.filter(split != "train")[0]
        error = np.abs(data.y.to_numpy()[mask] - prediction[mask])
        value = float(np.average(error, weights=data.weight.to_numpy()[mask]))
        return {f"{part}/mae": value}

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
        mask = split != "train"
        if not mask.any():
            return []
        part = split.filter(mask)[0]
        relative = Path(part) / "predictions.csv"
        destination = artifact_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {
                "timestamp": data.time.filter(mask),
                "observed": data.y.filter(mask),
                "predicted": prediction[mask.to_numpy()],
                "weight": data.weight.filter(mask),
            }
        ).write_csv(destination)
        return [relative]
