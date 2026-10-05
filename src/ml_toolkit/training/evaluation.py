"""Evaluate fitted models and log requested diagnostics directly to MLflow."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import Literal, Mapping

import mlflow
import numpy as np

from ..metrics.ranking import ranking_metrics
from ..metrics.pu import PUMetric
from ..plots import classification, monitoring, psi as psi_plots, pu as pu_plots, ranking, regression
from ..psi import PSIAnalysis, PSIConfig, PSIFeature
from ..tracking.mlflow import log_polars_table
from .contracts import TrainingData

logger = logging.getLogger(__name__)


def _feature_filename(prefix: str, index: int, feature: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", feature).strip("._")[:60] or "feature"
    return f"{prefix}_{index:03d}_{safe}.png"


@dataclass(frozen=True)
class DiagnosticsOptions:
    """Diagnostic choices supplied by a project, not a global toolkit config."""

    task: Literal["classification", "pu_classification", "regression", "frequency", "severity"]
    psi: bool = True
    plots: bool = True
    psi_bins: int = 20
    shap_top_n: int | None = None
    feature_plots: tuple[str, ...] | None = None
    ranking_selected_percents: tuple[float, ...] = (10.0,)

    def __post_init__(self) -> None:
        if self.task not in ("classification", "pu_classification", "regression", "frequency", "severity"):
            raise ValueError("task must be classification, pu_classification, regression, frequency, or severity")
        if self.psi_bins < 1:
            raise ValueError("psi_bins must be positive")
        if self.shap_top_n is not None and self.shap_top_n < 1:
            raise ValueError("shap_top_n must be positive")
        if not self.ranking_selected_percents or any(
            not 0 < percent <= 100 for percent in self.ranking_selected_percents
        ):
            raise ValueError("ranking_selected_percents must contain values in (0, 100]")


def _scores(metric, data: TrainingData, prediction: np.ndarray) -> dict[str, float]:
    p = np.asarray(prediction, dtype=float).reshape(-1)
    if p.shape != (len(data),) or not np.isfinite(p).all():
        raise ValueError("Predictions must contain one finite value per row")
    result = metric(data.y.to_numpy(), p, data.weight.to_numpy())
    if is_dataclass(result):
        values = asdict(result)
    elif isinstance(result, Mapping):
        values = result
    else:
        raise TypeError("metric must return a dataclass or mapping of metric names to values")
    return {str(key): float(value) for key, value in values.items()}


def log_evaluation(
    *, model, train: TrainingData, validation: TrainingData,
    train_prediction: np.ndarray, validation_prediction: np.ndarray,
    metric, diagnostics: DiagnosticsOptions | None, name: str,
) -> dict[str, dict[str, float]]:
    """Log split metrics, optional PSI and figures; return numeric scores."""
    pu_loss = getattr(model, "_pu_loss", None)
    if pu_loss is not None:
        if not isinstance(metric, PUMetric) or metric.loss != pu_loss or metric.prior != model._pu_prior:
            raise ValueError("PU models require PUMetric with the model's loss and prior")
        if diagnostics is not None and diagnostics.task != "pu_classification":
            raise ValueError("PU models require DiagnosticsOptions(task='pu_classification')")
    elif isinstance(metric, PUMetric) or (diagnostics is not None and diagnostics.task == "pu_classification"):
        raise ValueError("PU metrics and diagnostics require a PU model")
    mlflow.log_params({
        "train_n_rows": len(train), "train_date_start": str(train.time.min()),
        "train_date_end": str(train.time.max()), "val_n_rows": len(validation),
        "val_date_start": str(validation.time.min()),
        "val_date_end": str(validation.time.max()),
    })
    scores = {
        "train": _scores(metric, train, train_prediction),
        "val": _scores(metric, validation, validation_prediction),
    }
    for split, values in scores.items():
        mlflow.log_metrics({f"{split}/{key}": value for key, value in values.items()})
        logger.debug("Evaluation %s %s metrics: %s", name, split, values)
    if diagnostics is not None:
        _log_diagnostics(model, train, validation, train_prediction,
                         validation_prediction, diagnostics, name)
    return scores


def _log_diagnostics(model, train, validation, train_prediction, validation_prediction,
                     options: DiagnosticsOptions, name: str) -> None:
    if tuple(train.X.columns) != tuple(validation.X.columns):
        raise ValueError("Train and validation features must match for diagnostics")
    logger.debug(
        "Generating diagnostics for %s: task=%s, PSI=%s, plots=%s",
        name,
        options.task,
        options.psi,
        options.plots,
    )
    with TemporaryDirectory(prefix="ml-toolkit-diagnostics-") as directory:
        root = Path(directory)

        def emit(filename: str, draw, *args, **kwargs) -> None:
            path = root / filename
            draw(*args, save_fig=path, **kwargs)
            mlflow.log_artifact(str(path), artifact_path=f"figures/{name}")
            logger.debug("Logged diagnostic figure %s for %s", filename, name)

        psi_result = None
        if options.psi and train.X.columns:
            categorical = set(train.cat_features)
            features = tuple(PSIFeature(
                feature, "categorical" if (
                    feature in categorical or not train.X.schema[feature].is_numeric()
                ) else "numerical",
            ) for feature in train.X.columns)
            psi_result = PSIAnalysis(PSIConfig(features, n_bins=options.psi_bins)).fit(
                train.X, time=train.time,
            ).compare(validation.X, time=validation.time)
            log_polars_table(psi_result.psi, f"psi/{name}/scores.json")
            log_polars_table(psi_result.level_distribution,
                             f"psi/{name}/level_distribution.json")
            if not psi_result.monthly_distribution.is_empty():
                log_polars_table(psi_result.monthly_distribution,
                                 f"psi/{name}/monthly_distribution.json")

        if not options.plots:
            logger.debug("Plot diagnostics disabled for %s", name)
            return
        if psi_result is not None:
            emit("psi_scores.png", psi_plots.feature_psi_plot, psi_result.psi)
            for index, feature in enumerate(train.X.columns):
                if not psi_result.monthly_distribution.is_empty():
                    emit(_feature_filename("psi", index, feature), psi_plots.monthly_psi_level_distribution,
                         psi_result.monthly_distribution, feature)

        train_y = train.y.to_numpy()
        val_y = validation.y.to_numpy()
        train_weight = train.weight.to_numpy()
        val_weight = validation.weight.to_numpy()
        if options.task != "pu_classification":
            emit("monthly_observed_fitted.png", regression.monthly_observed_fitted,
                 train.time.to_list(), train_y, train_prediction,
                 validation.time.to_list(), val_y, validation_prediction,
                 reference_weight=train_weight, current_weight=val_weight,
                 reference_group=getattr(train, "business_type", None),
                 current_group=getattr(validation, "business_type", None))

        if options.task != "pu_classification" and np.all(val_y >= 0) and np.sum(val_y * val_weight) > 0:
            curve = ranking_metrics(val_y, validation_prediction, val_weight,
                                    options.ranking_selected_percents)
            emit("gains.png", ranking.gains_curve, curve)
            emit("lift.png", ranking.lift_curve, curve)
        if options.task == "classification":
            emit("precision_recall.png", classification.precision_recall,
                 val_y, validation_prediction)
            emit("f1_threshold.png", classification.f1_threshold,
                 val_y, validation_prediction)
            emit("calibration.png", classification.calibration,
                 val_y, validation_prediction)
        elif options.task == "pu_classification":
            emit("pu_score_distribution.png", pu_plots.score_distribution,
                 val_y, validation_prediction, weights=val_weight)
            emit("pu_positive_capture.png", pu_plots.positive_capture,
                 val_y, validation_prediction, weights=val_weight)

        if options.feature_plots is None:
            feature_names = tuple(feature for feature, dtype in train.X.schema.items()
                                  if dtype.is_numeric())
        else:
            feature_names = options.feature_plots
        time_column = "__ml_toolkit_plot_time__"
        while time_column in train.X.columns or time_column in validation.X.columns:
            time_column += "_"
        reference = train.X.with_columns(train.time.alias(time_column))
        current = validation.X.with_columns(validation.time.alias(time_column))
        for index, feature in enumerate(feature_names):
            if feature not in train.X.columns or not train.X.schema[feature].is_numeric():
                raise ValueError(f"Feature {feature!r} is missing or not numerical")
            emit(_feature_filename("feature", index, feature), monitoring.monthly_feature_distribution,
                 reference, current, feature, time_column)

        if options.shap_top_n is not None:
            from ..explainability import calculate_shap_values, top_shap_features
            from ..plots import explainability as shap_plots

            native = getattr(model, "native_model", model)
            explanation = calculate_shap_values(native, validation.X)
            emit("shap_importance.png", shap_plots.shap_importance, explanation,
                 max_features_display=options.shap_top_n)
            emit("shap_beeswarm.png", shap_plots.shap_beeswarm, explanation,
                 max_features_display=options.shap_top_n)
            for index, feature in enumerate(top_shap_features(explanation, options.shap_top_n)):
                emit(_feature_filename("shap", index, feature), shap_plots.shap_dependence,
                     explanation, feature)
