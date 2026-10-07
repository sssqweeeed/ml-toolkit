"""Evaluate fitted models and log requested diagnostics directly to MLflow."""

from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from numbers import Real
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import Mapping

import mlflow
import numpy as np

from ..config import (
    ClassificationPlotsConfig,
    DiagnosticsConfig,
    PUDiagnosticsConfig,
    PUPlotsConfig,
    PostFitEvaluationConfig,
    RegressionPlotsConfig,
)
from ..metrics.ranking import ranking_metrics
from ..metrics.pu import PUMetric
from ..plots import (
    classification,
    monitoring,
    psi as psi_plots,
    pu as pu_plots,
    ranking,
    regression,
)
from ..psi import PSIAnalysis, PSIConfig, PSIFeature
from ..tracking.mlflow import log_polars_table
from .contracts import (
    DatasetSplit,
    PostFitEvaluator,
    TrainingData,
    _validate_post_fit_evaluator,
)

logger = logging.getLogger(__name__)


def _feature_filename(prefix: str, index: int, feature: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", feature).strip("._")[:60] or "feature"
    return f"{prefix}_{index:03d}_{safe}.png"


def _scores(metric, data: TrainingData, prediction: np.ndarray) -> dict[str, float]:
    p = np.asarray(prediction, dtype=float)
    if p.shape != (len(data),) or not np.isfinite(p).all():
        raise ValueError("Predictions must contain one finite value per row")
    result = metric(data.y.to_numpy(), p, data.weight.to_numpy())
    if is_dataclass(result):
        values = asdict(result)
    elif isinstance(result, Mapping):
        values = result
    else:
        raise TypeError(
            "metric must return a dataclass or mapping of metric names to values"
        )
    scores = {}
    for key, value in values.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"Metric {key!r} must be a numeric scalar")
        scores[str(key)] = float(value)
    return scores


def _log_post_fit_evaluation(
    *,
    evaluator: PostFitEvaluator,
    data: TrainingData,
    prediction: np.ndarray,
    split: DatasetSplit,
    name: str,
    write_artifacts: bool = True,
) -> None:
    try:
        if not isinstance(split, DatasetSplit):
            raise TypeError("split must be a DatasetSplit member")
        _validate_post_fit_evaluator(data, evaluator)
        predicted = np.asarray(prediction, dtype=float)
        target = np.asarray(data.y.to_numpy(), dtype=float)
        if predicted.shape != (len(data),) or not np.isfinite(predicted).all():
            raise ValueError("Predictions must contain one finite value per row")
        if target.shape != (len(data),) or not np.isfinite(target).all():
            raise ValueError("Targets must contain one finite value per row")
        inputs = dict(
            object_id=getattr(data, "object_id"),
            y_pred=predicted,
            y_true=target,
            split=split,
        )
        values = evaluator.evaluate(**inputs)
        if not isinstance(values, dict):
            raise TypeError("post_fit_evaluator must return dict[str, float]")
        metrics = {}
        for key, value in values.items():
            if (
                not isinstance(key, str)
                or not re.fullmatch(r"[\w .\-/]+", key)
                or not key.strip()
                or any(part in ("", ".", "..") for part in key.split("/"))
            ):
                raise ValueError(f"Invalid post-fit metric name: {key!r}")
            if not isinstance(value, (float, np.floating)):
                raise TypeError(f"Post-fit metric {key!r} must be a floating scalar")
            numeric = float(value)
            if not np.isfinite(numeric):
                raise ValueError(f"Post-fit metric {key!r} must be finite")
            metrics[f"{split.value}/custom/{key}"] = numeric
        if metrics:
            mlflow.log_metrics(metrics)
            logger.debug(
                "Evaluation %s %s custom metrics: %s", name, split.value, metrics
            )
        writer = getattr(evaluator, "write_artifacts", None)
        if write_artifacts and writer is not None:
            _log_post_fit_artifacts(writer, inputs)
    except Exception as exc:
        exc.add_note(f"Post-fit evaluation stage={name}, split={split}")
        raise


def _log_post_fit_artifacts(writer, inputs) -> None:
    with TemporaryDirectory(prefix="ml-toolkit-post-fit-") as directory:
        root = Path(directory).resolve()
        paths = writer(**inputs, artifact_dir=root)
        if paths is None:
            return
        if not isinstance(paths, list):
            raise TypeError("write_artifacts must return list[Path] or None")
        images = []
        for path in paths:
            if not isinstance(path, Path):
                raise TypeError("write_artifacts paths must be pathlib.Path objects")
            image = (root / path).resolve()
            if not image.is_relative_to(root):
                raise ValueError("write_artifacts images must be inside artifact_dir")
            if not image.is_file():
                raise ValueError(f"write_artifacts image does not exist: {path}")
            if image.suffix.lower() not in {
                ".png",
                ".jpg",
                ".jpeg",
                ".svg",
                ".webp",
                ".gif",
                ".tif",
                ".tiff",
                ".bmp",
            }:
                raise ValueError(f"write_artifacts accepts only image files: {path}")
            images.append(image)
        for image in images:
            relative = image.relative_to(root)
            parent = relative.parent.as_posix()
            artifact_path = f"custom/{inputs['split'].value}"
            if parent != ".":
                artifact_path += f"/{parent}"
            mlflow.log_artifact(str(image), artifact_path=artifact_path)
            logger.debug("Logged post-fit image %s to %s", relative, artifact_path)


def _validate_task_metric(model, metric, diagnostics=None):
    pu_loss = model.pu_loss if getattr(model, "task", None) == "pu_classification" else None
    if pu_loss is not None:
        if (
            not isinstance(metric, PUMetric)
            or metric.loss != pu_loss
            or metric.prior != model.pu_prior
        ):
            raise ValueError(
                "PU models require PUMetric with the model's loss and prior"
            )
        if diagnostics is not None and not isinstance(diagnostics, PUDiagnosticsConfig):
            raise ValueError("PU models require PUDiagnosticsConfig")
    elif isinstance(metric, PUMetric) or (
        diagnostics is not None and isinstance(diagnostics, PUDiagnosticsConfig)
    ):
        raise ValueError("PU metrics and diagnostics require a PU model")


def log_evaluation(
    *,
    model,
    train: TrainingData,
    validation: TrainingData,
    train_prediction: np.ndarray,
    validation_prediction: np.ndarray,
    metric,
    diagnostics: DiagnosticsConfig | None,
    name: str,
    post_fit_evaluator: PostFitEvaluator | None = None,
    post_fit: PostFitEvaluationConfig | None = None,
) -> dict[str, dict[str, float]]:
    """Log split metrics, optional PSI and figures; return numeric scores."""
    _validate_task_metric(model, metric, diagnostics)
    mlflow.log_params(
        {
            "train_n_rows": len(train),
            "train_date_start": str(train.time.min()),
            "train_date_end": str(train.time.max()),
            "val_n_rows": len(validation),
            "val_date_start": str(validation.time.min()),
            "val_date_end": str(validation.time.max()),
        }
    )
    scores = {
        "train": _scores(metric, train, train_prediction),
        "val": _scores(metric, validation, validation_prediction),
    }
    for split, values in scores.items():
        mlflow.log_metrics({f"{split}/{key}": value for key, value in values.items()})
        logger.debug("Evaluation %s %s metrics: %s", name, split, values)
    post_fit = PostFitEvaluationConfig() if post_fit is None else post_fit
    if post_fit_evaluator is not None:
        for split, data, prediction in (
            (DatasetSplit.TRAIN, train, train_prediction),
            (DatasetSplit.VAL, validation, validation_prediction),
        ):
            if split.value not in post_fit.splits:
                continue
            _log_post_fit_evaluation(
                evaluator=post_fit_evaluator,
                data=data,
                prediction=prediction,
                split=split,
                name=name,
                write_artifacts=post_fit.write_artifacts,
            )
    if diagnostics is not None:
        _log_diagnostics(
            model,
            train,
            validation,
            train_prediction,
            validation_prediction,
            diagnostics,
            name,
        )
    return scores


def _diagnostics_requested(options: DiagnosticsConfig | None) -> bool:
    if options is None:
        return False
    if options.psi is not None:
        return True
    plots = options.plots
    if plots is None:
        return False
    common = plots.common
    if common.shap is not None or (
        common.numerical_distributions is not None
        and common.numerical_distributions.features != ()
    ):
        return True
    if isinstance(plots, PUPlotsConfig):
        return plots.score_distribution or plots.positive_capture
    if plots.gains or plots.lift or plots.monthly_observed_fitted:
        return True
    return isinstance(plots, ClassificationPlotsConfig) and (
        plots.precision_recall or plots.f1_threshold or plots.calibration
    )


def _log_diagnostics(
    model,
    train,
    validation,
    train_prediction,
    validation_prediction,
    options: DiagnosticsConfig,
    name: str,
) -> None:
    if tuple(train.X.columns) != tuple(validation.X.columns):
        raise ValueError("Train and validation features must match for diagnostics")
    logger.debug(
        "Generating diagnostics for %s: task=%s, PSI=%s, plots=%s",
        name,
        options.task,
        options.psi,
        options.plots,
    )
    plots = options.plots
    with TemporaryDirectory(prefix="ml-toolkit-diagnostics-") as directory:
        root = Path(directory)

        def emit(filename: str, draw, *args, **kwargs) -> None:
            figure = plots.common.figure
            path = (root / filename).with_suffix(f".{figure.format}")
            draw(*args, save_fig=path, figure=figure, **kwargs)
            mlflow.log_artifact(str(path), artifact_path="figures")
            logger.debug("Logged diagnostic figure %s for %s", path.name, name)

        psi_result = None
        if options.psi is not None and train.X.columns:
            categorical = set(train.cat_features)
            features = options.psi.features
            if features is None:
                features = tuple(
                    PSIFeature(
                        feature,
                        (
                            "categorical"
                            if feature in categorical
                            or not train.X.schema[feature].is_numeric()
                            else "numerical"
                        ),
                    )
                    for feature in train.X.columns
                )
            psi_result = (
                PSIAnalysis(
                    PSIConfig(
                        features,
                        n_bins=options.psi.n_bins,
                        epsilon=options.psi.epsilon,
                    )
                )
                .fit(train.X, time=train.time)
                .compare(validation.X, time=validation.time)
            )
            log_polars_table(psi_result.psi, "psi/scores.json")
            log_polars_table(
                psi_result.level_distribution, "psi/level_distribution.json"
            )
            if not psi_result.monthly_distribution.is_empty():
                log_polars_table(
                    psi_result.monthly_distribution,
                    "psi/monthly_distribution.json",
                )
        if plots is None:
            logger.debug("Plot diagnostics disabled for %s", name)
            return
        common = plots.common
        if psi_result is not None:
            if common.psi_scores:
                emit("psi_scores.png", psi_plots.feature_psi_plot, psi_result.psi)
            psi_names = (
                tuple(psi_result.psi["feature"].to_list())
                if common.psi_distribution_features is None
                else common.psi_distribution_features
            )
            if not set(psi_names) <= set(psi_result.psi["feature"].to_list()):
                raise ValueError(
                    "PSI distribution features must belong to the PSI configuration"
                )
            if not psi_result.monthly_distribution.is_empty():
                for index, feature in enumerate(psi_names):
                    emit(
                        _feature_filename("psi", index, feature),
                        psi_plots.monthly_psi_level_distribution,
                        psi_result.monthly_distribution,
                        feature,
                    )

        train_y, val_y = train.y.to_numpy(), validation.y.to_numpy()
        train_weight, val_weight = train.weight.to_numpy(), validation.weight.to_numpy()
        if isinstance(plots, (ClassificationPlotsConfig, RegressionPlotsConfig)):
            if plots.monthly_observed_fitted:
                emit(
                    "monthly_observed_fitted.png",
                    regression.monthly_observed_fitted,
                    train.time.to_list(),
                    train_y,
                    train_prediction,
                    validation.time.to_list(),
                    val_y,
                    validation_prediction,
                    reference_weight=train_weight,
                    current_weight=val_weight,
                    reference_group=getattr(train, "business_type", None),
                    current_group=getattr(validation, "business_type", None),
                )
            if plots.gains or plots.lift:
                if not np.all(val_y >= 0) or np.sum(val_y * val_weight) <= 0:
                    raise ValueError(
                        "Enabled gains/lift require nonnegative targets with positive weighted total"
                    )
                curve = ranking_metrics(val_y, validation_prediction, val_weight)
                if plots.gains:
                    emit("gains.png", ranking.gains_curve, curve)
                if plots.lift:
                    emit("lift.png", ranking.lift_curve, curve)
        if isinstance(plots, ClassificationPlotsConfig):
            for enabled, filename, draw in (
                (
                    plots.precision_recall,
                    "precision_recall.png",
                    classification.precision_recall,
                ),
                (plots.f1_threshold, "f1_threshold.png", classification.f1_threshold),
                (plots.calibration, "calibration.png", classification.calibration),
            ):
                if enabled:
                    emit(filename, draw, val_y, validation_prediction, config=plots)
        elif isinstance(plots, PUPlotsConfig):
            if plots.score_distribution:
                emit(
                    "pu_score_distribution.png",
                    pu_plots.score_distribution,
                    val_y,
                    validation_prediction,
                    weights=val_weight,
                    config=plots,
                )
            if plots.positive_capture:
                emit(
                    "pu_positive_capture.png",
                    pu_plots.positive_capture,
                    val_y,
                    validation_prediction,
                    weights=val_weight,
                )

        distribution = common.numerical_distributions
        if distribution is not None:
            feature_names = distribution.features
            if feature_names is None:
                feature_names = tuple(
                    feature
                    for feature, dtype in train.X.schema.items()
                    if dtype.is_numeric()
                )
            time_column = "__ml_toolkit_plot_time__"
            while time_column in train.X.columns or time_column in validation.X.columns:
                time_column += "_"
            reference = train.X.with_columns(train.time.alias(time_column))
            current = validation.X.with_columns(validation.time.alias(time_column))
            for index, feature in enumerate(feature_names):
                if (
                    feature not in train.X.columns
                    or not train.X.schema[feature].is_numeric()
                ):
                    raise ValueError(f"Feature {feature!r} is missing or not numerical")
                emit(
                    _feature_filename("feature", index, feature),
                    monitoring.monthly_feature_distribution,
                    reference,
                    current,
                    feature,
                    time_column,
                    config=distribution,
                )
        if common.shap is not None:
            from ..explainability import calculate_shap_values
            from ..plots import explainability as shap_plots

            shap = common.shap
            if not set(shap.dependence_features) <= set(validation.X.columns):
                raise ValueError("SHAP dependence features must exist in the input")
            explanation = calculate_shap_values(
                getattr(model, "native_model", model), validation.X, config=shap
            )
            emit(
                "shap_importance.png",
                shap_plots.shap_importance,
                explanation,
                config=shap,
            )
            emit(
                "shap_beeswarm.png",
                shap_plots.shap_beeswarm,
                explanation,
                config=shap,
            )
            for index, feature in enumerate(shap.dependence_features):
                emit(
                    _feature_filename("shap", index, feature),
                    shap_plots.shap_dependence,
                    explanation,
                    feature,
                )
