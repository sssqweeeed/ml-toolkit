"""Portable PSI reference containing fitted rules and aggregate counts only."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import cast

import mlflow
import mlflow.artifacts
import polars as pl
from mlflow.models import Model

from ..feature_levels import (
    LevelTransformer,
    decode_feature_scalar,
    encode_feature_scalar,
    json_object,
)
from ..types import FloatArray
from .analysis import LEVEL_SCHEMA, TIME_SCHEMA, resolve_time
from .score import PSIConfig, PSIResult, psi_from_shares

COUNT_SCHEMA = {"feature": pl.String, "level": pl.String, "rows": pl.Int64}


def _counts(transformer: LevelTransformer, levels: pl.DataFrame) -> pl.DataFrame:
    assert transformer.feature_levels is not None
    tables: list[pl.DataFrame] = []
    for feature in transformer.feature_levels:
        values = levels.get_column(feature.config.name).cast(pl.String)
        if values.null_count() or not values.is_in(feature.levels).all():
            raise ValueError(
                f"Feature {feature.config.name!r} contains undefined levels"
            )
        counts = (
            values.to_frame("level")
            .group_by("level")
            .agg(pl.len().cast(pl.Int64).alias("rows"))
        )
        tables.append(
            pl.DataFrame(
                {"level": list(feature.levels)},
                schema={"level": pl.String},
            )
            .with_row_index("level_order")
            .join(counts, on="level", how="left")
            .sort("level_order")
            .with_columns(
                pl.lit(feature.config.name).alias("feature"),
                pl.col("rows").fill_null(0),
            )
            .select(*COUNT_SCHEMA)
        )
    return pl.concat(tables)


@dataclass(frozen=True)
class PSIBaseline:
    """A fitted transformer and unweighted reference counts, without source rows.

    Use ``from_reference`` to fit, ``save``/``load`` for local persistence, and
    ``compare`` for each monitoring window. Time metadata describes the reference;
    comparisons return an empty time_distribution rather than refitting levels.
    """

    transformer: LevelTransformer
    reference_counts: pl.DataFrame
    metadata: dict[str, object]

    @classmethod
    def from_reference(
        cls,
        reference: pl.DataFrame,
        *,
        config: PSIConfig,
        time: pl.Series | None = None,
        stage: str | None = None,
    ) -> PSIBaseline:
        """Fit on all reference rows; weights and serving-example limits do not apply."""
        if reference.is_empty():
            raise ValueError("PSI reference must not be empty")
        time = resolve_time(reference, time)
        transformer = LevelTransformer(config.features, n_bins=config.n_bins).fit(
            reference
        )
        metadata: dict[str, object] = {
            "version": 1,
            "toolkit_version": version("ml-toolkit"),
            "epsilon": config.epsilon,
            "n_rows": len(reference),
            "stage": stage,
            "time_start": encode_feature_scalar(
                None if time is None else cast(date | datetime, time.min())
            ),
            "time_end": encode_feature_scalar(
                None if time is None else cast(date | datetime, time.max())
            ),
        }
        return cls(
            transformer,
            _counts(transformer, transformer.transform(reference)),
            metadata,
        )

    def _validate(self) -> None:
        metadata = self.metadata
        if type(metadata.get("version")) is not int or metadata["version"] != 1:
            raise ValueError("Unsupported PSI baseline format version")
        rows = metadata.get("n_rows")
        if type(rows) is not int or rows < 1:
            raise ValueError("Invalid PSI reference row count")
        epsilon = metadata.get("epsilon")
        if type(epsilon) not in (int, float) or not 0 < cast(float, epsilon) < 1:
            raise ValueError("Invalid PSI epsilon")
        if not isinstance(metadata.get("toolkit_version"), str):
            raise ValueError("Missing toolkit version")
        if metadata.get("stage") is not None and not isinstance(metadata["stage"], str):
            raise ValueError("Invalid training stage")
        start = decode_feature_scalar(metadata["time_start"])
        end = decode_feature_scalar(metadata["time_end"])
        if (start is None) != (end is None) or (
            start is not None
            and (
                not isinstance(start, date) or not isinstance(end, date) or start > end
            )
        ):
            raise ValueError("Invalid reference time range")
        features = self.transformer.feature_levels
        if not features:
            raise ValueError("PSI baseline must have fitted features")
        counts = self.reference_counts
        if counts.schema != pl.Schema(COUNT_SCHEMA) or any(counts.null_count().row(0)):
            raise ValueError("Invalid reference counts schema or null values")
        expected = [(f.config.name, level) for f in features for level in f.levels]
        actual = list(counts.select("feature", "level").iter_rows())
        if actual != expected or len(set(actual)) != len(actual):
            raise ValueError("Reference counts do not match fitted features and levels")
        if (counts["rows"] < 0).any() or any(
            total != rows
            for total in counts.group_by("feature").agg(pl.col("rows").sum())["rows"]
        ):
            raise ValueError("Reference counts must be nonnegative and sum to n_rows")

    def save(self, directory: str | Path) -> None:
        """Write a versioned three-file baseline, preserving exact fitted state."""
        self._validate()
        payload = self.transformer.to_dict()
        LevelTransformer.from_dict(payload)
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        (root / "manifest.json").write_text(
            json.dumps(self.metadata, indent=2, allow_nan=False), encoding="utf-8"
        )
        (root / "levels.json").write_text(
            json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
        )
        self.reference_counts.write_parquet(root / "reference_counts.parquet")

    @classmethod
    def load(cls, directory: str | Path) -> PSIBaseline:
        """Restore and validate an aggregate baseline without fitting or source data."""
        root = Path(directory)
        try:
            metadata = json_object(
                json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            )
            levels_payload = json_object(
                json.loads((root / "levels.json").read_text(encoding="utf-8"))
            )
            for stored in cast(list[object], levels_payload["features"]):
                stored_levels = cast(list[object], json_object(stored)["levels"])
                if len(set(stored_levels)) != len(stored_levels):
                    raise ValueError(
                        "PSI baseline was saved with ambiguous interval labels "
                        "(merged bins) and cannot be loaded; re-create it with "
                        "PSIBaseline.from_reference"
                    )
            transformer = LevelTransformer.from_dict(levels_payload)
            result = cls(
                transformer,
                pl.read_parquet(root / "reference_counts.parquet"),
                metadata,
            )
            result._validate()
            return result
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Incomplete PSI baseline at {root}: {exc.filename}"
            ) from exc
        except (ValueError, TypeError, KeyError, pl.exceptions.PolarsError) as exc:
            raise ValueError(f"Invalid PSI baseline at {root}: {exc}") from exc

    def compare(self, current: pl.DataFrame) -> PSIResult:
        """Compare a nonempty batch against fixed levels and reference frequencies."""
        self._validate()
        if current.is_empty():
            raise ValueError("PSI current data must not be empty")
        levels = self.transformer.transform(current)
        current_counts = _counts(self.transformer, levels)
        reference = self.reference_counts.with_columns(
            pl.lit("reference").alias("sample"),
            (pl.col("rows") / cast(int, self.metadata["n_rows"])).alias("share"),
        )
        current_counts = current_counts.with_columns(
            pl.lit("current").alias("sample"),
            (pl.col("rows") / len(current)).alias("share"),
        )
        features = self.transformer.feature_levels
        assert features is not None
        distributions: list[pl.DataFrame] = []
        scores: list[float] = []
        for feature in features:
            ref = reference.filter(pl.col("feature") == feature.config.name)
            cur = current_counts.filter(pl.col("feature") == feature.config.name)
            scores.append(
                psi_from_shares(
                    cast(FloatArray, ref["share"].to_numpy()),
                    cast(FloatArray, cur["share"].to_numpy()),
                    epsilon=cast(float, self.metadata["epsilon"]),
                )
            )
            grid = pl.DataFrame({"level": feature.levels}).with_row_index("level_order")
            distributions.extend(
                grid.join(table, on="level", how="left")
                .sort("level_order")
                .select(*LEVEL_SCHEMA)
                for table in (ref, cur)
            )
        return PSIResult(
            psi=pl.DataFrame(
                {"feature": [f.config.name for f in features], "psi": scores}
            ),
            level_distribution=pl.concat(distributions),
            time_distribution=pl.DataFrame(schema=TIME_SCHEMA),
            feature_levels=features,
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
