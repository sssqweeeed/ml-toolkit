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
from ..feature_levels import FeatureLevelConfig, LevelTransformer
from ..metrics.ranking import ranking_metrics
from ..models.base import Model
from ..plots import (
    classification,
    monitoring,
)
from ..plots import psi as psi_plots
from ..plots import pu as pu_plots
from ..plots import ranking, regression
from ..psi import compare_psi
from ..tracking import runs as tracking
from ..tracking.mlflow import log_polars_table
from ..types import FloatArray
from .contracts import (
    DatasetSplit,
    PredictionEvaluator,
    Pool,
)


class EvaluationInputs(TypedDict):
    object_id: pl.Series
    y_pred: FloatArray
    y_true: FloatArray
    split: DatasetSplit


logger = logging.getLogger(__name__)


def _feature_filename(index: int, feature: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", feature).strip("._")[:60] or "feature"
    return f"{index:03d}_{safe}.png"


def report_custom_evaluation(
    *,
    evaluator: PredictionEvaluator,
    data: Pool,
    prediction: FloatArray,
    split: DatasetSplit,
    name: str,
    write_artifacts: bool = True,
) -> None:
    try:
        if data.object_id is None:
            raise ValueError(
                "object_id is required when prediction_evaluators is supplied"
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
                raise ValueError(f"Invalid prediction evaluation metric name: {key!r}")
            numeric = float(value)
            if not np.isfinite(numeric):
                raise ValueError(f"Prediction evaluation metric {key!r} must be finite")
            metrics[f"{split.value}/{evaluator.name}/{key}"] = numeric
        if metrics:
            tracking.log_metrics(metrics)
            logger.debug(
                "Evaluation %s %s custom metrics: %s", name, split.value, metrics
            )
        if write_artifacts:
            report_custom_artifacts(evaluator, inputs)
    except Exception as exc:
        exc.add_note(
            f"Prediction evaluation evaluator={evaluator.name}, "
            f"stage={name}, split={split}"
        )
        raise


def report_custom_artifacts(
    evaluator: PredictionEvaluator, inputs: EvaluationInputs
) -> None:
    with TemporaryDirectory(prefix="ml-toolkit-prediction-evaluation-") as directory:
        root = Path(directory).resolve()
        paths = evaluator.write_artifacts(**inputs, artifact_dir=root)
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
            artifact_path = f"{evaluator.name}/{inputs['split'].value}"
            if parent != ".":
                artifact_path += f"/{parent}"
            tracking.log_artifact(image, artifact_path=artifact_path)
            logger.debug(
                "Logged prediction evaluation artifact %s to %s", relative, artifact_path
            )


def report_diagnostics(
    model: Model,
    train: Pool,
    validation: Pool,
    train_prediction: FloatArray,
    validation_prediction: FloatArray,
    options: DiagnosticsConfig,
    name: str,
    split: DatasetSplit = DatasetSplit.VAL,
) -> None:
    if tuple(train.X.columns) != tuple(validation.X.columns):
        raise ValueError("Train and validation features must match for diagnostics")
    logger.debug(
        "Generating diagnostics for %s: task=%s, plots=%s",
        name,
        options.task,
        options.plots,
    )
    plots = options.plots
    if plots is None:
        logger.debug("Plot diagnostics disabled for %s", name)
        return
    common = plots.common
    with TemporaryDirectory(prefix="ml-toolkit-diagnostics-") as directory:
        root = Path(directory)

        def emit(
            artifact_file: str,
            draw: Callable[..., object],
            *args: object,
            **kwargs: object,
        ) -> None:
            assert plots is not None
            figure = plots.common.figure
            relative = Path(artifact_file).with_suffix(f".{figure.format}")
            path = root / relative
            draw(*args, save_fig=path, figure=figure, **kwargs)
            tracking.log_artifact(path, artifact_path=relative.parent.as_posix())
            logger.debug(
                "Logged diagnostic figure %s for %s", relative.as_posix(), name
            )

        train_y, val_y = train.y.to_numpy(), validation.y.to_numpy()
        train_weight, val_weight = train.weight.to_numpy(), validation.weight.to_numpy()
        if common.psi or common.observed_fitted:
            features = common.features
            if features is None:
                categorical = set(train.cat_features)
                features = tuple(
                    FeatureLevelConfig(
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
            if features:
                transformer = LevelTransformer(features, n_bins=common.n_bins).fit(
                    train.X
                )
                reference_levels = transformer.transform(train.X)
                current_levels = transformer.transform(validation.X)
                feature_levels = transformer.feature_levels
                assert feature_levels is not None
                if common.psi:
                    psi_result = compare_psi(
                        reference_levels,
                        current_levels,
                        feature_levels,
                        epsilon=common.epsilon,
                        period=common.period,
                        reference_time=train.time,
                        current_time=validation.time,
                    )
                    log_polars_table(
                        psi_result.psi, "tables/psi/scores.json"
                    )
                    log_polars_table(
                        psi_result.level_distribution,
                        "tables/psi/level_distribution.json",
                    )
                    if not psi_result.time_distribution.is_empty():
                        log_polars_table(
                            psi_result.time_distribution,
                            "tables/psi/time_distribution.json",
                        )
                    emit(
                        "figures/psi/scores.png",
                        psi_plots.feature_psi_plot,
                        psi_result.psi,
                    )
                    if not psi_result.time_distribution.is_empty():
                        for index, feature in enumerate(feature_levels):
                            emit(
                                "figures/psi/time_by_feature/"
                                f"{_feature_filename(index, feature.config.name)}",
                                psi_plots.time_psi_level_distribution,
                                psi_result.time_distribution,
                                feature.config.name,
                                period=common.period,
                            )
                if common.observed_fitted:
                    for index, feature in enumerate(feature_levels):
                        emit(
                            "figures/observed_fitted/by_feature/"
                            f"{_feature_filename(index, feature.config.name)}",
                            regression.observed_fitted_by_level,
                            reference_levels.get_column(feature.config.name),
                            train_y,
                            train_prediction,
                            current_levels.get_column(feature.config.name),
                            val_y,
                            validation_prediction,
                            level_order=feature.levels,
                            reference_weight=train_weight,
                            current_weight=val_weight,
                            feature_name=feature.config.name,
                            current_label=(
                                "Test" if split == DatasetSplit.TEST else "Validation"
                            ),
                        )

        if isinstance(plots, (ClassificationPlotsConfig, RegressionPlotsConfig)):
            if plots.time_observed_fitted:
                emit(
                    "figures/observed_fitted/time.png",
                    regression.time_observed_fitted,
                    train.time.to_list(),
                    train_y,
                    train_prediction,
                    validation.time.to_list(),
                    val_y,
                    validation_prediction,
                    reference_weight=train_weight,
                    current_weight=val_weight,
                    period=common.period,
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
                    emit(
                        f"figures/performance/{split.value}/gains.png",
                        ranking.gains_curve,
                        curve,
                    )
                if plots.lift:
                    emit(
                        f"figures/performance/{split.value}/lift.png",
                        ranking.lift_curve,
                        curve,
                    )
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
                    emit(
                        f"figures/performance/{split.value}/{filename}",
                        draw,
                        val_y,
                        validation_prediction,
                        weight=val_weight,
                        config=plots,
                    )
        elif isinstance(plots, PUPlotsConfig):
            if plots.score_distribution:
                emit(
                    f"figures/performance/{split.value}/score_distribution.png",
                    pu_plots.score_distribution,
                    val_y,
                    validation_prediction,
                    weights=val_weight,
                    config=plots,
                )
            if plots.positive_capture:
                emit(
                    f"figures/performance/{split.value}/positive_capture.png",
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
                    "figures/feature_distributions/time_by_feature/"
                    f"{_feature_filename(index, feature)}",
                    monitoring.time_feature_distribution,
                    reference,
                    current,
                    feature,
                    time_column,
                    period=common.period,
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
                f"figures/shap/{split.value}/importance.png",
                shap_plots.shap_importance,
                explanation,
                config=shap,
            )
            emit(
                f"figures/shap/{split.value}/beeswarm.png",
                shap_plots.shap_beeswarm,
                explanation,
                config=shap,
            )
            for index, feature in enumerate(shap.dependence_features):
                emit(
                    f"figures/shap/{split.value}/dependence/"
                    f"{_feature_filename(index, feature)}",
                    shap_plots.shap_dependence,
                    explanation,
                    feature,
                )
