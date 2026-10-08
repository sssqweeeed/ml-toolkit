"""Shared validation/test diagnostics and custom reporting."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypedDict

import numpy as np
import polars as pl

from ..config import (
    ClassificationPlotsConfig,
    DiagnosticsConfig,
    PUPlotsConfig,
    RegressionPlotsConfig,
)
from ..metrics.ranking import ranking_metrics
from ..models.base import Model
from ..plots import (
    classification,
    monitoring,
)
from ..plots import psi as psi_plots
from ..plots import pu as pu_plots
from ..plots import ranking, regression
from ..psi import PSIAnalysis, PSIConfig, PSIFeature
from ..tracking import runs as tracking
from ..tracking.mlflow import log_polars_table
from ..types import FloatArray
from .contracts import (
    DatasetSplit,
    PostFitEvaluator,
    TrainingData,
)


class EvaluationInputs(TypedDict):
    object_id: pl.Series
    y_pred: FloatArray
    y_true: FloatArray
    split: DatasetSplit


logger = logging.getLogger(__name__)


def _feature_filename(prefix: str, index: int, feature: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", feature).strip("._")[:60] or "feature"
    return f"{prefix}_{index:03d}_{safe}.png"


def report_custom_evaluation(
    *,
    evaluator: PostFitEvaluator,
    data: TrainingData,
    prediction: FloatArray,
    split: DatasetSplit,
    name: str,
    write_artifacts: bool = True,
) -> None:
    try:
        if data.object_id is None:
            raise ValueError(
                "object_id is required when post_fit_evaluator is supplied"
            )
        inputs: EvaluationInputs = {
            "object_id": data.object_id,
            "y_pred": prediction,
            "y_true": data.y.to_numpy().astype(float),
            "split": split,
        }
        values = evaluator.evaluate(**inputs)
        metrics: dict[str, float] = {}
        for key, value in values.items():
            if (
                not re.fullmatch(r"[\w .\-/]+", key)
                or not key.strip()
                or any(part in ("", ".", "..") for part in key.split("/"))
            ):
                raise ValueError(f"Invalid post-fit metric name: {key!r}")
            numeric = float(value)
            if not np.isfinite(numeric):
                raise ValueError(f"Post-fit metric {key!r} must be finite")
            metrics[f"{split.value}/custom/{key}"] = numeric
        if metrics:
            tracking.log_metrics(metrics)
            logger.debug(
                "Evaluation %s %s custom metrics: %s", name, split.value, metrics
            )
        if write_artifacts:
            report_custom_artifacts(evaluator, inputs)
    except Exception as exc:
        exc.add_note(f"Post-fit evaluation stage={name}, split={split}")
        raise


def report_custom_artifacts(
    evaluator: PostFitEvaluator, inputs: EvaluationInputs
) -> None:
    with TemporaryDirectory(prefix="ml-toolkit-post-fit-") as directory:
        root = Path(directory).resolve()
        paths = evaluator.write_artifacts(**inputs, artifact_dir=root)
        images: list[Path] = []
        for path in paths:
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
            tracking.log_artifact(image, artifact_path=artifact_path)
            logger.debug("Logged post-fit image %s to %s", relative, artifact_path)


def report_diagnostics(
    model: Model,
    train: TrainingData,
    validation: TrainingData,
    train_prediction: FloatArray,
    validation_prediction: FloatArray,
    options: DiagnosticsConfig,
    name: str,
    split: DatasetSplit = DatasetSplit.VAL,
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

        def emit(
            filename: str, draw: Callable[..., object], *args: object, **kwargs: object
        ) -> None:
            assert plots is not None
            figure = plots.common.figure
            path = (root / filename).with_suffix(f".{figure.format}")
            draw(*args, save_fig=path, figure=figure, **kwargs)
            tracking.log_artifact(path, artifact_path=f"{split.value}/figures")
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
            log_polars_table(psi_result.psi, f"{split.value}/psi/scores.json")
            log_polars_table(
                psi_result.level_distribution,
                f"{split.value}/psi/level_distribution.json",
            )
            if not psi_result.monthly_distribution.is_empty():
                log_polars_table(
                    psi_result.monthly_distribution,
                    f"{split.value}/psi/monthly_distribution.json",
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
                    reference_group=train.business_type,
                    current_group=validation.business_type,
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
                model.native_model, validation.X, config=shap
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
