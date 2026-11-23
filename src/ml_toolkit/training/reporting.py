"""Execute additional reporting and upload its explicitly selected artifacts."""

from collections.abc import Callable, Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol

import mlflow
import polars as pl

from .._console import DatasetRow, Logger, datasets
from ..config import ReportingConfig
from ..data import Pool, validate_split_labels
from ..explainability import (
    calculate_shap_values,
    shap_beeswarm,
    shap_dependence,
    shap_importance,
)
from ..feature_levels import LevelTransformer, feature_definitions
from ..feature_selection import (
    CustomSelectionSettings,
    FeatureSelectionResult,
    ForwardSettings,
    RFESettings,
    TopKSettings,
)
from ..feature_selection import figures as selection_figures
from ..figures import FigureConfig, feature_filename
from ..metrics import Metric
from ..models import TrackingConfig
from ..models.base import Model
from ..psi import PSIBaseline, PSIConfig, compare_psi
from ..psi import figures as psi_plots
from ..reports import (
    DataAnalyzer,
    ModelEvaluator,
    PredictionEvaluator,
    comparison_pair,
    data_comparison_pair,
    validate_inputs,
    validate_report_metrics,
)
from ..types import FloatArray

logger = Logger(__name__)


class _SelectionFigure(Protocol):
    def __call__(
        self,
        result: FeatureSelectionResult,
        save_fig: Path,
        *,
        figure: FigureConfig | None = None,
    ) -> None: ...


def print_datasets(title: str, samples: Mapping[str, Pool]) -> None:
    """Print the row, feature and date-range summary of named samples."""
    datasets(
        title,
        [
            DatasetRow(
                name, len(pool), len(pool.X.columns), pool.time[0], pool.time[-1]
            )
            for name, pool in samples.items()
        ],
    )


def prepare_evaluators(
    evaluators: PredictionEvaluator | list[PredictionEvaluator] | None,
    config: ReportingConfig,
    analyzers: tuple[DataAnalyzer, ...] = (),
    model_evaluators: tuple[ModelEvaluator, ...] = (),
    objective_evaluator: PredictionEvaluator | None = None,
) -> tuple[PredictionEvaluator, ...]:
    """Normalize and validate names before fitting, even for disabled stages."""
    items = (
        ()
        if evaluators is None
        else tuple(evaluators)
        if isinstance(evaluators, list)
        else (evaluators,)
    )
    names: set[str] = set()
    reserved: set[str] = set()
    if config.psi:
        reserved.add("psi")
    if config.shap is not None:
        reserved.add("shap")
    checked = items
    if objective_evaluator is not None and not any(
        item is objective_evaluator for item in items
    ):
        checked = (*items, objective_evaluator)
    for item in (*checked, *model_evaluators, *analyzers):
        name = item.name.casefold()
        if name in names or name in reserved:
            raise ValueError(
                f"Evaluator and analyzer names must be unique and not reserved: {item.name!r}"
            )
        names.add(name)
    return items


def prepare_model_evaluators(
    evaluators: ModelEvaluator | list[ModelEvaluator] | None,
) -> tuple[ModelEvaluator, ...]:
    """Normalize fitted-model reports independently of prediction reports."""
    return (
        ()
        if evaluators is None
        else tuple(evaluators)
        if isinstance(evaluators, list)
        else (evaluators,)
    )


def prepare_analyzers(
    analyzers: DataAnalyzer | list[DataAnalyzer] | None,
) -> tuple[DataAnalyzer, ...]:
    """Normalize data reports independently of prediction evaluators."""
    return (
        ()
        if analyzers is None
        else tuple(analyzers)
        if isinstance(analyzers, list)
        else (analyzers,)
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
        mlflow.log_artifact(str(image), artifact_path=artifact_path)
        logger.debug("Logged report artifact %s to %s", relative, artifact_path)


def report_feature_selection(
    result: FeatureSelectionResult, config: ReportingConfig, *, stage: str
) -> None:
    """Plot a known selector's single search history in its active run."""
    settings = result.settings
    reports: list[tuple[str, _SelectionFigure]]
    match settings:
        case RFESettings():
            reports = [("rfe_progress", selection_figures.rfe_progress)]
        case TopKSettings():
            reports = [("top_k_ranking", selection_figures.top_k_ranking)]
        case ForwardSettings():
            reports = [
                ("forward_progress", selection_figures.forward_progress),
                ("forward_improvement", selection_figures.forward_improvement),
            ]
        case CustomSelectionSettings() | None:
            return
    reports.append(("best_iteration", selection_figures.best_iteration_progress))
    try:
        with TemporaryDirectory(prefix="ml-toolkit-selection-") as directory:
            root = Path(directory).resolve()
            paths: list[Path] = []
            for index, (name, draw) in enumerate(reports):
                path = Path(f"{name}.{config.figure.format}")
                draw(result, root / path, figure=config.figure)
                # Additional diagnostics may have no comparable/available data.
                if index == 0 or (root / path).is_file():
                    paths.append(path)
            upload_artifacts(root, paths, "selection")
    except Exception as exc:
        exc.add_note(f"Feature selection selector={settings.selector}, stage={stage}")
        raise


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
    model_evaluators: tuple[ModelEvaluator, ...] = (),
    metric: Metric | None = None,
    psi_baseline: PSIBaseline | None = None,
    scored_evaluator: PredictionEvaluator | None = None,
) -> None:
    """Run one already-enabled fit's reports; stage is only error context."""
    validate_inputs(data, prediction, split)
    if model_evaluators and metric is None:
        raise ValueError("metric is required for model evaluators")
    for evaluator in evaluators:
        _report_prediction(
            evaluator,
            data=data,
            prediction=prediction,
            split=split,
            config=config,
            stage=stage,
            scored=evaluator is scored_evaluator,
        )
    if metric is not None:
        for model_evaluator in model_evaluators:
            _report_model(
                model_evaluator,
                model=model,
                metric=metric,
                data=data,
                split=split,
                config=config,
                stage=stage,
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


def _report_prediction(
    evaluator: PredictionEvaluator,
    *,
    data: Pool,
    prediction: FloatArray,
    split: pl.Series,
    config: ReportingConfig,
    stage: str,
    scored: bool,
) -> None:
    """Run one prediction evaluator; a scored one only writes artifacts."""
    _run_report(
        name=evaluator.name,
        evaluate=(
            None
            if scored
            else lambda: evaluator.evaluate(
                data=data, prediction=prediction, split=split
            )
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


def _report_model(
    evaluator: ModelEvaluator,
    *,
    model: Model,
    metric: Metric,
    data: Pool,
    split: pl.Series,
    config: ReportingConfig,
    stage: str,
) -> None:
    _run_report(
        name=evaluator.name,
        evaluate=lambda: evaluator.evaluate(
            model=model, metric=metric, data=data, split=split
        ),
        write=lambda root: evaluator.write_artifacts(
            model=model,
            metric=metric,
            data=data,
            split=split,
            artifact_dir=root,
            figure=config.figure,
        ),
        context=f"Model evaluation evaluator={evaluator.name}, stage={stage}",
    )


def _report_analyzer(
    analyzer: DataAnalyzer,
    *,
    data: Pool,
    split: pl.Series,
    config: ReportingConfig,
    stage: str,
) -> None:
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


def evaluate_report_metrics(
    *,
    name: str,
    evaluate: Callable[[], dict[str, float]],
    context: str,
) -> dict[str, float]:
    try:
        metrics = validate_report_metrics(evaluate())
        if metrics:
            mlflow.log_metrics(
                {f"{name}/{key}": value for key, value in metrics.items()}
            )
        return metrics
    except Exception as exc:
        exc.add_note(context)
        raise


def _run_report(
    *,
    name: str,
    evaluate: Callable[[], dict[str, float]] | None,
    write: Callable[[Path], list[Path]],
    context: str,
) -> None:
    """Apply the same metric and artifact rules to both report interfaces."""
    if evaluate is not None:
        evaluate_report_metrics(name=name, evaluate=evaluate, context=context)
    try:
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
    validate_split_labels(data, split)
    for analyzer in analyzers:
        _report_analyzer(analyzer, data=data, split=split, config=config, stage=stage)
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
        else LevelTransformer(features, n_bins=config.n_bins).fit(
            train.to_analysis_frame()
        )
    )
    assert transformer.feature_levels is not None
    result = compare_psi(
        transformer.transform(train.to_analysis_frame()),
        transformer.transform(current.to_analysis_frame()),
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
    shap = config.shap
    assert shap is not None
    if not set(shap.dependence_features) <= set(current.X.columns):
        raise ValueError("SHAP dependence features must exist in the input")
    explanation = calculate_shap_values(model.native_model, current.X, config=shap)
    for name, draw in (
        ("importance", shap_importance),
        ("beeswarm", shap_beeswarm),
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
        shap_dependence(explanation, feature, save_fig=path, figure=config.figure)
        paths.append(path)
    return paths


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
