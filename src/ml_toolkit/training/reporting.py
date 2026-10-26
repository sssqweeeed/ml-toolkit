"""Execute additional reporting and upload its explicitly selected artifacts."""

import logging
from collections.abc import Callable
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl

from ..config import ReportingConfig
from ..data import Pool
from ..data_analysis import DataAnalyzer
from .._reporting import (
    data_comparison_pair,
    validate_split,
    feature_definitions,
    feature_filename,
)
from ..evaluation import (
    PredictionEvaluator,
    comparison_pair,
    validate_inputs,
)
from ..feature_levels import LevelTransformer
from ..models.base import Model
from ..plots import psi as psi_plots
from ..psi import PSIBaseline, compare_psi
from ..tracking import runs as tracking
from ..types import FloatArray

logger = logging.getLogger(__name__)


def prepare_evaluators(
    evaluators: PredictionEvaluator | list[PredictionEvaluator] | None,
    config: ReportingConfig,
    analyzers: tuple[DataAnalyzer, ...] = (),
) -> tuple[PredictionEvaluator, ...]:
    """Normalize and validate names before fitting, even for disabled stages."""
    items = (
        ()
        if evaluators is None
        else tuple(evaluators) if isinstance(evaluators, list) else (evaluators,)
    )
    names: set[str] = set()
    reserved: set[str] = set()
    if config.psi:
        reserved.add("psi")
    if config.shap is not None:
        reserved.add("shap")
    for item in (*items, *analyzers):
        name = item.name.casefold()
        if name in names or name in reserved:
            raise ValueError(
                f"Evaluator and analyzer names must be unique and not reserved: {item.name!r}"
            )
        names.add(name)
    return items


def prepare_analyzers(
    analyzers: DataAnalyzer | list[DataAnalyzer] | None,
) -> tuple[DataAnalyzer, ...]:
    """Normalize data reports independently of prediction evaluators."""
    return (
        ()
        if analyzers is None
        else tuple(analyzers) if isinstance(analyzers, list) else (analyzers,)
    )


def upload_artifacts(root: Path, paths: list[Path], namespace: str) -> None:
    """Validate every selected path before uploading any files."""
    files: list[Path] = []
    for path in paths:
        image = root / path
        resolved = image.resolve()
        if (
            not image.is_relative_to(root)
            or ".." in image.relative_to(root).parts
            or not resolved.is_relative_to(root)
        ):
            raise ValueError("write_artifacts files must be inside artifact_dir")
        if not image.is_file():
            raise ValueError(f"write_artifacts file does not exist: {path}")
        if image.suffix.lower() not in {
            ".pdf",
            ".png",
            ".jpg",
            ".jpeg",
            ".svg",
            ".webp",
            ".gif",
            ".tif",
            ".tiff",
            ".bmp",
            ".csv",
            ".parquet",
            ".json",
            ".html",
        }:
            raise ValueError(
                f"write_artifacts does not support this file format: {path}"
            )
        files.append(image)
    for image in files:
        relative = image.relative_to(root)
        parent = relative.parent.as_posix()
        artifact_path = namespace
        if parent != ".":
            artifact_path += f"/{parent}"
        tracking.log_artifact(image, artifact_path=artifact_path)
        logger.debug("Logged report artifact %s to %s", relative, artifact_path)


def report_predictions(
    *,
    model: Model,
    data: Pool,
    prediction: FloatArray,
    split: pl.Series,
    evaluators: tuple[PredictionEvaluator, ...],
    config: ReportingConfig,
    stage: str,
    analyzers: tuple[DataAnalyzer, ...] = (),
    psi_baseline: PSIBaseline | None = None,
) -> None:
    """Run one already-enabled fit's reports; stage is only error context."""
    validate_inputs(data, prediction, split)
    for evaluator in evaluators:
        _run_report(
            name=evaluator.name,
            evaluate=lambda: evaluator.evaluate(
                data=data, prediction=prediction, split=split
            ),
            write=lambda root: evaluator.write_artifacts(
                data=data,
                prediction=prediction,
                split=split,
                artifact_dir=root,
                figure=config.figure,
                features=config.features,
                n_bins=config.n_bins,
            ),
            context=f"Prediction evaluation evaluator={evaluator.name}, stage={stage}",
        )
    report_data(
        data=data,
        split=split,
        analyzers=analyzers,
        config=config,
        stage=stage,
        psi_baseline=psi_baseline,
    )
    if config.shap is not None:
        try:
            with TemporaryDirectory(prefix="ml-toolkit-shap-") as directory:
                root = Path(directory).resolve()
                upload_artifacts(
                    root,
                    _shap_artifacts(model, data, prediction, split, root, config),
                    "shap",
                )
        except Exception as exc:
            exc.add_note(f"Prediction evaluation evaluator=shap, stage={stage}")
            raise


def _run_report(
    *,
    name: str,
    evaluate: Callable[[], dict[str, float]],
    write: Callable[[Path], list[Path]],
    context: str,
) -> None:
    """Apply the same metric and artifact rules to both report interfaces."""
    try:
        values = evaluate()
        metrics: dict[str, float] = {}
        for key, value in values.items():
            if (
                not re.fullmatch(r"[\w .\-/]+", key)
                or not key.strip()
                or any(part in ("", ".", "..") for part in key.split("/"))
            ):
                raise ValueError(f"Invalid report metric name: {key!r}")
            numeric = float(value)
            if not np.isfinite(numeric):
                raise ValueError(f"Report metric {key!r} must be finite")
            metrics[f"{name}/{key}"] = numeric
        if metrics:
            tracking.log_metrics(metrics)
        with TemporaryDirectory(prefix="ml-toolkit-report-") as directory:
            root = Path(directory).resolve()
            upload_artifacts(root, write(root), name)
    except Exception as exc:
        exc.add_note(context)
        raise


def report_data(
    *,
    data: Pool,
    split: pl.Series,
    analyzers: tuple[DataAnalyzer, ...],
    config: ReportingConfig,
    stage: str,
    psi_baseline: PSIBaseline | None = None,
) -> None:
    """Run data analysis and PSI without a model or predictions."""
    validate_split(data, split)
    for analyzer in analyzers:
        _run_report(
            name=analyzer.name,
            evaluate=lambda: analyzer.evaluate(data=data, split=split),
            write=lambda root: analyzer.write_artifacts(
                data=data,
                split=split,
                artifact_dir=root,
                figure=config.figure,
                features=config.features,
                n_bins=config.n_bins,
            ),
            context=f"Data analysis analyzer={analyzer.name}, stage={stage}",
        )
    if config.psi:
        try:
            with TemporaryDirectory(prefix="ml-toolkit-psi-") as directory:
                root = Path(directory).resolve()
                upload_artifacts(
                    root, _psi_artifacts(data, split, root, config, psi_baseline), "psi"
                )
        except Exception as exc:
            exc.add_note(f"Data analysis analyzer=psi, stage={stage}")
            raise


def _psi_artifacts(
    data: Pool,
    split: pl.Series,
    root: Path,
    config: ReportingConfig,
    psi_baseline: PSIBaseline | None = None,
) -> list[Path]:
    paths: list[Path] = []
    pair = data_comparison_pair(data, split)
    if pair is None:
        return []
    _, train, current = pair
    features = feature_definitions(train, config.features)
    if not features:
        return []
    transformer = (
        psi_baseline.transformer
        if psi_baseline is not None
        else LevelTransformer(features, n_bins=config.n_bins).fit(train.X)
    )
    assert transformer.feature_levels is not None
    result = compare_psi(
        transformer.transform(train.X),
        transformer.transform(current.X),
        transformer.feature_levels,
        epsilon=config.epsilon,
        period=config.period,
        reference_time=train.time,
        current_time=current.time,
    )
    root.mkdir(parents=True, exist_ok=True)
    for name, table in (
        ("scores", result.psi),
        ("level_distribution", result.level_distribution),
        ("time_distribution", result.time_distribution),
    ):
        if table.is_empty():
            continue
        path = root / f"{name}.json"
        table.write_json(path)
        paths.append(path)
    path = root / f"scores.{config.figure.format}"
    psi_plots.feature_psi_plot(result.psi, save_fig=path, figure=config.figure)
    paths.append(path)
    if not result.time_distribution.is_empty():
        for index, feature in enumerate(features):
            path = (
                root
                / "time_by_feature"
                / feature_filename(index, feature.name, config.figure.format)
            )
            psi_plots.time_psi_level_distribution(
                result.time_distribution,
                feature.name,
                save_fig=path,
                period=config.period,
                figure=config.figure,
            )
            paths.append(path)
    return paths


def _shap_artifacts(
    model: Model,
    data: Pool,
    prediction: FloatArray,
    split: pl.Series,
    root: Path,
    config: ReportingConfig,
) -> list[Path]:
    paths: list[Path] = []
    pair = comparison_pair(data, prediction, split)
    if pair is None:
        return []
    part, _, _, current, _ = pair
    from ..explainability import calculate_shap_values
    from ..plots import explainability as shap_plots

    shap = config.shap
    assert shap is not None
    if not set(shap.dependence_features) <= set(current.X.columns):
        raise ValueError("SHAP dependence features must exist in the input")
    explanation = calculate_shap_values(model.native_model, current.X, config=shap)
    for name, draw in (
        ("importance", shap_plots.shap_importance),
        ("beeswarm", shap_plots.shap_beeswarm),
    ):
        path = root / part / f"{name}.{config.figure.format}"
        draw(explanation, save_fig=path, config=shap, figure=config.figure)
        paths.append(path)
    for index, feature in enumerate(shap.dependence_features):
        path = (
            root
            / part
            / "dependence"
            / feature_filename(index, feature, config.figure.format)
        )
        shap_plots.shap_dependence(
            explanation, feature, save_fig=path, figure=config.figure
        )
        paths.append(path)
    return paths
