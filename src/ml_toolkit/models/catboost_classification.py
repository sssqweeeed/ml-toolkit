from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, cast

import catboost
import mlflow
import numpy as np
import pandas as pd
import polars as pl
from mlflow.models.model import ModelInfo
from mlflow.pyfunc.model import PythonModel
from mlflow.pyfunc.utils import pyfunc

from ..metrics._pu import risk_components
from ..tracking._model_state import log_catboost
from ..tracking.state import ClassificationState
from ..types import FloatArray, IntArray, ModelParams
from ._catboost import (
    feature_importances,
    native_fit,
    normalize_loss,
    prepare_frame,
    read_state,
    refit_parameters,
)
from ._catboost_upu import UPUObjective, upu_risk

if TYPE_CHECKING:
    from ..config import TrackingConfig
    from ..training.contracts import TrainingData

logger = logging.getLogger("ml_toolkit.models.catboost_classification")


def _pu_risks(
    logits: FloatArray,
    labels: FloatArray,
    weights: FloatArray,
    prior: float,
) -> tuple[float, float]:
    """Return the positive and estimated negative logistic PU risks."""
    logits = np.asarray(logits, dtype=float).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if not (logits.shape == labels.shape == weights.shape):
        raise ValueError("PU logits, labels, and weights must have equal lengths")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("PU labels must be 1 (labeled positive) or 0 (unlabeled)")
    if not np.isfinite(logits).all() or not np.isfinite(weights).all():
        raise ValueError("PU logits and weights must be finite")
    if (weights < 0).any():
        raise ValueError("PU weights must be non-negative")
    if not 0 < prior < 1:
        raise ValueError("pu_prior must be strictly between 0 and 1")
    positive = labels == 1
    unlabeled = ~positive
    if weights[positive].sum() <= 0 or weights[unlabeled].sum() <= 0:
        raise ValueError("PU training needs positive-weight rows in both groups")

    return risk_components(
        np.logaddexp(0.0, -logits), np.logaddexp(0.0, logits), labels, weights, prior
    )


class _PURiskMetric:
    """Evaluate PU risk over the complete CatBoost training or eval pool."""

    def __init__(self, prior: float, *, non_negative: bool) -> None:
        self.prior = prior
        self.non_negative = non_negative

    def is_max_optimal(self) -> bool:
        return False

    def get_final_error(self, error: float, weight: float) -> float:
        return error / weight

    def evaluate(
        self,
        approxes: Sequence[Sequence[float]],
        target: Sequence[float],
        weight: Sequence[float] | None,
    ) -> tuple[float, float]:
        if len(approxes) != 1:
            raise ValueError("PU risk requires one-dimensional raw predictions")
        weights = np.ones(len(target)) if weight is None else weight
        positive_risk, negative_risk = _pu_risks(
            np.asarray(approxes[0]), np.asarray(target), np.asarray(weights), self.prior
        )
        risk = positive_risk + (
            max(0.0, negative_risk) if self.non_negative else negative_risk
        )
        return risk, 1.0


class UPUMetric(_PURiskMetric):
    no_jit = True

    def __init__(self, prior: float) -> None:
        super().__init__(prior, non_negative=False)

    def evaluate(
        self,
        approxes: Sequence[Sequence[float]],
        target: Sequence[float],
        weight: Sequence[float] | None,
    ) -> tuple[float, float]:
        if len(approxes) != 1:
            raise ValueError("PU risk requires one-dimensional raw predictions")
        risk = upu_risk(
            np.asarray(approxes[0], dtype=np.float64),
            np.asarray(target),
            None if weight is None else np.asarray(weight),
            self.prior,
        )
        return risk, 1.0


class NNPUMetric(_PURiskMetric):
    def __init__(self, prior: float) -> None:
        super().__init__(prior, non_negative=True)


class _PULogisticObjective:
    """Per-row derivatives for the current uPU or nnPU gradient branch."""

    def __init__(
        self,
        prior: float,
        positive_scale: float,
        unlabeled_scale: float,
        *,
        reverse_negative_risk: bool = False,
    ) -> None:
        self.prior = prior
        self.positive_scale = positive_scale
        self.unlabeled_scale = unlabeled_scale
        self.reverse_negative_risk = reverse_negative_risk

    def calc_ders_range(
        self,
        approxes: Sequence[float],
        targets: Sequence[float],
        weights: Sequence[float] | None,
    ) -> list[tuple[float, float]]:
        result: list[tuple[float, float]] = []
        for index in range(len(targets)):
            logit = float(approxes[index])
            if logit >= 0:
                probability = 1.0 / (1.0 + math.exp(-logit))
            else:
                exp_logit = math.exp(logit)
                probability = exp_logit / (1.0 + exp_logit)
            weight = 1.0 if weights is None else float(weights[index])
            curvature = probability * (1.0 - probability)

            if targets[index] == 1:
                if self.reverse_negative_risk:
                    # -R_negative has derivative +prior * sigmoid(logit).
                    der1 = -self.positive_scale * probability
                    der2 = -self.positive_scale * curvature
                else:
                    # Logistic uPU: loss_+(logit) - loss_-(logit) = -logit.
                    der1 = self.positive_scale
                    der2 = 0.0
            elif targets[index] == 0:
                direction = 1.0 if self.reverse_negative_risk else -1.0
                der1 = direction * self.unlabeled_scale * probability
                der2 = direction * self.unlabeled_scale * curvature
            else:
                raise ValueError("PU labels must be 1 or 0")
            result.append((weight * der1, weight * der2))
        return result


class CatBoostClassificationModel(PythonModel):
    """Train and serve a binary CatBoost classifier with weighted log-odds.

    Parameters
    ----------
    params : dict or None, default None
        Copied CatBoost parameters; loss accepts Logloss or CrossEntropy.
        ``decision_threshold`` fixes the label cutoff and disables F1 tuning.
        Otherwise validation tunes unweighted F1, or the cutoff defaults to 0.5.
    """

    def __init__(self, params: ModelParams | None = None) -> None:
        self.params = {} if params is None else params.copy()
        self._model: catboost.CatBoostClassifier | None = None
        self._baseline: float | None = None
        self._threshold: float | None = None
        self._validation_f1: float | None = None
        self._pu_loss: str | None = None
        self._pu_prior: float | None = None
        self._best_iteration: int | None = None

    @property
    def native_model(self) -> catboost.CatBoostClassifier:
        """Return the fitted native CatBoost model; raise RuntimeError before fitting."""
        if self._model is None:
            raise RuntimeError("Model has not been fitted")
        return self._model

    @property
    def baseline(self) -> float:
        """Return the fitted log-odds baseline; raise RuntimeError before fitting."""
        if self._baseline is None:
            raise RuntimeError("Model has not been fitted")
        return self._baseline

    @property
    def threshold(self) -> float:
        """Return the hard-label probability cutoff.

        Returns
        -------
        float
            Supervised F1-selected cutoff, 0.5 without validation, or explicitly
            supplied decision_threshold.

        Raises
        ------
        RuntimeError
            The model is unfitted or a PU model has no explicit decision threshold.
        """
        if self._model is None:
            raise RuntimeError("Model has not been fitted")
        if self._threshold is None:
            raise RuntimeError(
                "PU label prediction requires an explicit decision_threshold"
            )
        return self._threshold

    @property
    def validation_f1(self) -> float | None:
        """Return the unweighted positive-class F1 used for threshold tuning, if available.

        Notes
        -----
        None for PU models and supervised fits without validation.
        """
        return self._validation_f1

    @property
    def best_iteration(self) -> int | None:
        """Return the zero-based best iteration index, or None when unavailable."""
        if self._best_iteration is not None:
            return self._best_iteration
        value = self.native_model.get_best_iteration()
        if value is None or value < 0:
            return None
        return int(value)

    def feature_importances(self, data: TrainingData) -> dict[str, float]:
        """Rank the current fitted feature columns.

        Parameters
        ----------
        data : TrainingData
            Rows with the fitted feature names in order.

        Returns
        -------
        dict[str, float]
            Native prediction-change importance per feature.
        """
        return feature_importances(self.native_model, data.X.columns)

    def get_refit_params(self) -> ModelParams:
        """Return copied constructor parameters for final fitting.

        Returns
        -------
        ModelParams
            The outer-validation best iteration plus one, when available;
            otherwise the original iteration budget. Early stopping is removed
            and use_best_model is disabled. Learned baselines are recalculated.
            Includes the selected decision threshold when one exists.

        Raises
        ------
        RuntimeError
            The adapter has not been fitted.
        """
        self.native_model
        params = refit_parameters(self.params, self.best_iteration)
        if self._threshold is not None:
            params["decision_threshold"] = self._threshold
        return params

    @property
    def feature_names(self) -> list[str]:
        """Return fitted feature names in native model order."""
        names = self.native_model.feature_names_
        if not names:
            raise RuntimeError("Fitted model does not contain feature names")
        return list(names)

    @property
    def cat_features(self) -> list[str]:
        """Return categorical feature names in native model order."""
        return [
            self.feature_names[index]
            for index in self.native_model.get_cat_feature_indices()
        ]

    @property
    def numerical_features(self) -> list[str]:
        """Return fitted feature names not marked categorical."""
        categorical = set(self.cat_features)
        return [name for name in self.feature_names if name not in categorical]

    @property
    def task(self) -> str:
        """Return the supervised binary classification task identifier."""
        return "classification"

    def fit(self, train: TrainingData, validation: TrainingData | None = None) -> Self:
        """Fit the classifier with the subclass's baseline and training policy.

        Parameters
        ----------
        train : TrainingData
            Aligned features, labels and exposure weights.
        validation : TrainingData or None, default None
            Optional holdout for early stopping and supervised F1 tuning.

        Returns
        -------
        Self
            Fitted adapter, resetting earlier threshold and iteration state.
        """
        logger.debug(
            "Fitting CatBoost classifier: %s training rows, validation=%s",
            len(train),
            validation is not None,
        )
        params = normalize_loss(self.params)
        threshold = params.pop("decision_threshold", None)
        if threshold is not None:
            try:
                threshold = float(cast(float, threshold))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "decision_threshold must be a number between 0 and 1"
                ) from error
            if not np.isfinite(threshold) or not 0 <= threshold <= 1:
                raise ValueError(
                    "decision_threshold must be finite and between 0 and 1"
                )
        self._validation_f1 = None
        self._best_iteration = None
        self._validate_params(params, train)
        baseline = self._calculate_baseline(train)
        params["boost_from_average"] = False
        self._model = self._fit_backend(train, validation, baseline, params)
        self._baseline = baseline
        self._configure_threshold(validation, threshold)
        logger.debug(
            "Fitted CatBoost classifier: PU loss=%s, threshold=%s",
            self._pu_loss,
            self._threshold,
        )
        return self

    def _validate_params(self, params: ModelParams, train: TrainingData) -> None:
        loss = params.setdefault("loss_function", "Logloss")
        if not isinstance(loss, str) or loss.split(":", 1)[0] not in (
            "Logloss",
            "CrossEntropy",
        ):
            raise ValueError("Binary classification requires Logloss or CrossEntropy")
        if "pu_prior" in params:
            raise ValueError("pu_prior requires CatBoostPUClassificationModel")
        y = train.y.to_numpy()
        if not np.isfinite(y).all() or (y < 0).any() or (y > 1).any():
            raise ValueError("Binary classification targets must be between 0 and 1")
        if loss.split(":", 1)[0] == "Logloss" and not np.isin(y, [0, 1]).all():
            raise ValueError("Logloss targets must be binary")

    def _fit_backend(
        self,
        train: TrainingData,
        validation: TrainingData | None,
        baseline: float,
        params: ModelParams,
    ) -> catboost.CatBoostClassifier:
        return native_fit(
            catboost.CatBoostClassifier,
            params,
            train.to_catboost(baseline=baseline),
            None if validation is None else validation.to_catboost(baseline=baseline),
        )

    def _configure_threshold(
        self, validation: TrainingData | None, threshold: float | None
    ) -> None:
        self._threshold = 0.5 if threshold is None else threshold
        if validation is not None and threshold is None:
            self._threshold, self._validation_f1 = self._select_threshold(
                validation.y.to_numpy(),
                self.predict_pool(validation),
            )

    def predict_pool(self, data: TrainingData) -> FloatArray:
        """Predict positive-class probabilities for project data.

        Parameters
        ----------
        data : TrainingData
            Project data whose X contains all fitted model features.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Positive-class probabilities, including the fitted log-odds baseline.

        Raises
        ------
        RuntimeError
            The model has not been fitted.
        """
        return self.predict_frame(data.to_model_input())

    def predict_labels(self, data: pl.DataFrame | pd.DataFrame) -> IntArray:
        """Predict labels using probability >= the stored threshold.

        Parameters
        ----------
        data : polars.DataFrame or pandas.DataFrame
            Feature frame containing all fitted feature names.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Integer labels 0 and 1.

        Raises
        ------
        RuntimeError
            The model is unfitted or PU label prediction lacks decision_threshold.
        """
        return (self.predict_frame(data) >= self.threshold).astype(np.int64)

    def predict_labels_pool(self, data: TrainingData) -> IntArray:
        """Predict thresholded binary labels for project-owned data.

        Parameters
        ----------
        data : TrainingData
            Project data with fitted feature columns in X.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Integer labels produced using the stored threshold.

        Raises
        ------
        RuntimeError
            The model is unfitted or a PU model lacks decision_threshold.
        """
        return self.predict_labels(data.to_model_input())

    @pyfunc
    def predict(
        self, context: Any, model_input: pd.DataFrame, params: ModelParams | None = None
    ) -> FloatArray:
        """Predict through the MLflow PythonModel interface.

        Parameters
        ----------
        context : Any
            MLflow context; unused.
        model_input : pandas.DataFrame
            Serving frame containing fitted feature names.
        params : dict or None, default None
            MLflow inference parameters; unused.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Positive-class probabilities, including the fitted log-odds baseline.

        See Also
        --------
        predict_frame : Predict directly from a Polars or pandas feature frame.
        """
        return self.predict_frame(model_input)

    def log(
        self,
        sample: TrainingData,
        *,
        config: TrackingConfig | None = None,
    ) -> ModelInfo:
        """Package and log the fitted model with a serving input example.

        Parameters
        ----------
        sample : TrainingData
            Serving example rows, capped by config.input_example_rows.
        config : TrackingConfig or None, default None
            Artifact name, positive example-row limit, registry name, and serving
            environment. None uses TrackingConfig defaults.

        Returns
        -------
        mlflow.models.ModelInfo
            Logged model information, including its model URI.

        Notes
        -----
        Writes native model and configuration artifacts, infers a serving
        signature, and logs baseline and input-schema metadata to MLflow. Configure
        tracking and an experiment before calling. Classification artifacts also
        preserve threshold and PU settings.
        """
        from ..config import TrackingConfig

        config = TrackingConfig() if config is None else config
        name = config.model_name
        registered_model_name = config.registered_model_name
        conda_env = config.conda_env
        sample_frame = sample.to_model_input(n_rows=config.input_example_rows).loc[
            :, self.feature_names
        ]
        model_info = self._log_artifacts(
            sample_frame,
            self.predict_frame(sample_frame),
            name=name,
            registered_model_name=registered_model_name,
            conda_env=conda_env,
        )
        mlflow.log_param("baseline", self.baseline)
        if self._threshold is not None:
            mlflow.log_param("threshold", self._threshold)
        if self.validation_f1 is not None:
            mlflow.log_metric("validation_f1_at_threshold", self.validation_f1)
        mlflow.log_dict(
            {
                "numerical_features": self.numerical_features,
                "cat_features": self.cat_features,
            },
            "model_metadata/catboost_input_schema.json",
        )
        return model_info

    @staticmethod
    def _select_threshold(
        y_true: FloatArray, probabilities: FloatArray
    ) -> tuple[float, float]:
        """Maximize unweighted binary F1; prefer the highest threshold on ties."""
        y_true = np.asarray(y_true)
        probabilities = np.asarray(probabilities, dtype=float)
        if (
            y_true.ndim != 1
            or probabilities.ndim != 1
            or y_true.size == 0
            or y_true.shape != probabilities.shape
        ):
            raise ValueError(
                "Labels and probabilities must be matching nonempty vectors"
            )
        if not np.isin(y_true, [0, 1]).all():
            raise ValueError("Threshold tuning requires binary labels 0 and 1")
        if (
            not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or (probabilities > 1).any()
        ):
            raise ValueError("Probabilities must be finite and between 0 and 1")

        # Without positive examples, every threshold has F1=0
        # under the zero_division=0 convention. Keep the default.
        if not (y_true == 1).any():
            return 0.5, 0.0

        # Sort once and evaluate each distinct score as a threshold.
        # Group tied probabilities so this matches the >= prediction rule.
        order = np.argsort(-probabilities, kind="stable")
        sorted_probabilities = probabilities[order]
        cumulative_tp = np.cumsum(y_true[order] == 1)
        group_ends = np.r_[
            np.flatnonzero(np.diff(sorted_probabilities) != 0),
            y_true.size - 1,
        ]
        true_positives = cumulative_tp[group_ends]
        predicted_positives = group_ends + 1
        actual_positives = cumulative_tp[-1]
        # F1 = 2*TP / (2*TP + FP + FN).
        scores = 2.0 * true_positives / (predicted_positives + actual_positives)
        # Thresholds are descending: argmax chooses the highest on ties.
        best_index = int(np.argmax(scores))
        threshold = sorted_probabilities[group_ends[best_index]]
        return float(cast(float, threshold)), float(scores[best_index])

    def _calculate_baseline(self, train: TrainingData) -> float:
        """Calculate the baseline log-odds from the positive class prevalence."""
        weights = train.weight.to_numpy()
        y_values = train.y.to_numpy()
        weighted_sum = np.average(y_values, weights=weights)
        if not np.isfinite(weighted_sum):
            raise ValueError("Weighted target mean must be finite")
        epsilon = 1e-7
        weighted_sum = np.clip(weighted_sum, epsilon, 1 - epsilon)
        return float(np.log(weighted_sum / (1 - weighted_sum)))

    def predict_frame(self, data: pl.DataFrame | pd.DataFrame) -> FloatArray:
        """Predict positive-class probabilities, adding fitted log-odds once.

        Parameters
        ----------
        data : polars.DataFrame or pandas.DataFrame
            Features selected and reordered to the fitted native schema.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Positive-class probabilities. PU scores need not be calibrated.
        """
        logger.debug("Predicting with CatBoost classifier: %s rows", len(data))
        frame = prepare_frame(data, self.feature_names, self.cat_features)
        logits = (
            np.asarray(
                self.native_model.predict(frame, prediction_type="RawFormulaVal"),
                dtype=float,
            )
            + self.baseline
        )
        return np.exp(-np.logaddexp(0.0, -logits))

    def _log_artifacts(
        self,
        sample_frame: pd.DataFrame,
        sample_prediction: FloatArray,
        *,
        name: str,
        registered_model_name: str | None,
        conda_env: Path | None,
    ) -> ModelInfo:
        return log_catboost(
            self,
            sample_frame,
            sample_prediction,
            name=name,
            registered_model_name=registered_model_name,
            conda_env=conda_env,
        )

    def export_state(self) -> ClassificationState:
        return {
            "version": 1,
            "model_type": type(self).__name__,
            "prediction_type": "Probability",
            "baseline": self.baseline,
            "baseline_mode": "raw_prediction",
            "threshold": self._threshold,
            "validation_f1": self._validation_f1,
            "pu_loss": self._pu_loss,
            "pu_prior": self._pu_prior,
            "best_iteration": self._best_iteration,
            "training_params": self.params.copy(),
        }

    def load_context(self, context: Any) -> None:
        state = cast(
            ClassificationState,
            read_state(context, type(self).__name__, "raw_prediction"),
        )
        self._model = catboost.CatBoostClassifier()
        self._model.load_model(context.artifacts["model"])
        self.params = state["training_params"]
        self._baseline = state["baseline"]
        self._threshold = state["threshold"]
        self._validation_f1 = state["validation_f1"]
        self._pu_loss, self._pu_prior = state["pu_loss"], state["pu_prior"]
        self._best_iteration = state["best_iteration"]
        logger.debug("Loaded CatBoost classifier: PU loss=%s", self._pu_loss)


class CatBoostPUClassificationModel(CatBoostClassificationModel):
    """Train positive-unlabeled CatBoost with a population-prior baseline.

    Parameters
    ----------
    params : dict or None, default None
        Copied parameters: ``loss`` is uPU (default) or nnPU; ``pu_prior`` must
        be finite and strictly between zero and one. uPU supports CPU and a
        single CUDA GPU; nnPU requires CPU. Both label groups need positive
        total weight, and class-weight options are not supported.
        Labels mean known positive (1) and unlabeled (0). Hard-label prediction
        requires an explicit ``decision_threshold``; F1 is never tuned.

    Notes
    -----
    uPU uses Numba-compiled derivatives and risk on CPU. With task_type='GPU',
    derivatives run in a CUDA kernel; risk evaluation remains on CPU and
    transfers predictions from the GPU. CUDA training requires a compatible
    CUDA toolkit and Numba CUDA support. The first fit compiles the kernels.
    nnPU evaluates the complete training risk before each tree, reversing the
    negative-risk gradient when needed (beta=0, gamma=1).

    Examples
    --------
    >>> model = CatBoostPUClassificationModel(
    ...     {"loss": "uPU", "pu_prior": 0.1, "task_type": "CPU"})
    >>> gpu_model = CatBoostPUClassificationModel(
    ...     {"loss": "uPU", "pu_prior": 0.1, "task_type": "GPU", "devices": "0"})
    """

    @property
    def task(self) -> str:
        """Return the positive-unlabeled classification task identifier."""
        return "pu_classification"

    @property
    def pu_loss(self) -> str:
        """Return the fitted PU risk policy, uPU or nnPU."""
        self.native_model
        assert self._pu_loss is not None
        return self._pu_loss

    @property
    def pu_prior(self) -> float:
        """Return the fitted positive population prior."""
        self.native_model
        assert self._pu_prior is not None
        return self._pu_prior

    def _validate_params(self, params: ModelParams, train: TrainingData) -> None:
        loss = params.pop("loss_function", "uPU")
        if loss not in ("uPU", "nnPU"):
            raise ValueError("PU classification requires uPU or nnPU loss")
        if "pu_prior" not in params:
            raise ValueError("PU loss requires pu_prior in params")
        prior = float(cast(float, params.pop("pu_prior")))
        if not np.isfinite(prior) or not 0 < prior < 1:
            raise ValueError("pu_prior must be finite and strictly between 0 and 1")
        task_type = str(params.get("task_type", "CPU")).upper()
        if task_type not in ("CPU", "GPU"):
            raise ValueError("PU task_type must be 'CPU' or 'GPU'")
        if loss == "nnPU" and task_type != "CPU":
            raise ValueError("nnPU requires task_type='CPU'")
        params["task_type"] = task_type
        if task_type == "GPU":
            # PU normalizations describe one complete pool, not GPU shards.
            device = str(params.setdefault("devices", "0"))
            if not device.isdecimal():
                raise ValueError(
                    "uPU GPU training requires one device, e.g. devices='0'"
                )
            params["devices"] = device
        for name in ("class_weights", "auto_class_weights", "scale_pos_weight"):
            if name in params:
                raise ValueError(f"{name} is incompatible with PU loss")
        self._pu_loss, self._pu_prior = loss, prior
        _pu_risks(
            np.zeros(len(train)),
            train.y.to_numpy(),
            train.weight.to_numpy(),
            prior,
        )

    def _calculate_baseline(self, train: TrainingData) -> float:
        assert self._pu_prior is not None
        return float(np.log(self._pu_prior / (1.0 - self._pu_prior)))

    def _fit_backend(
        self,
        train: TrainingData,
        validation: TrainingData | None,
        baseline: float,
        params: ModelParams,
    ) -> catboost.CatBoostClassifier:
        assert self._pu_prior is not None
        labels, weights = train.y.to_numpy(), train.weight.to_numpy()
        positive_weight = weights[labels == 1].sum()
        unlabeled_weight = weights[labels == 0].sum()
        total = positive_weight + unlabeled_weight
        positive_scale = float(self._pu_prior * total / positive_weight)
        unlabeled_scale = float(total / unlabeled_weight)
        params.update(
            leaf_estimation_method="Gradient",
            leaf_estimation_iterations=1,
            boosting_type="Plain",
        )
        metric = params.get("eval_metric", self._pu_loss)
        if isinstance(metric, str) and metric in ("uPU", "nnPU"):
            metric_class = UPUMetric if metric == "uPU" else NNPUMetric
            params["eval_metric"] = metric_class(self._pu_prior)
        if validation is not None:
            _pu_risks(
                np.full(len(validation), baseline),
                validation.y.to_numpy(),
                validation.weight.to_numpy(),
                self._pu_prior,
            )
        if self._pu_loss == "nnPU":
            objective = _PULogisticObjective(
                self._pu_prior, positive_scale, unlabeled_scale
            )
            model, self._best_iteration = self._fit_nnpu(
                train=train,
                validation=validation,
                train_data=train.to_catboost(baseline=baseline),
                baseline=baseline,
                prior=self._pu_prior,
                objective=objective,
                params=params,
            )
            return model
        if params["task_type"] == "GPU":
            from ._catboost_upu_cuda import make_upu_gpu_objective

            params["loss_function"] = make_upu_gpu_objective(
                positive_scale, unlabeled_scale
            )
        else:
            params["loss_function"] = UPUObjective(positive_scale, unlabeled_scale)
        if validation is not None:
            params.setdefault("use_best_model", True)
        return super()._fit_backend(train, validation, baseline, params)

    def _configure_threshold(
        self, validation: TrainingData | None, threshold: float | None
    ) -> None:
        self._threshold = threshold

    @staticmethod
    def _fit_nnpu(
        *,
        train: TrainingData,
        validation: TrainingData | None,
        train_data: catboost.Pool,
        baseline: float,
        prior: float,
        objective: _PULogisticObjective,
        params: ModelParams,
    ) -> tuple[catboost.CatBoostClassifier, int | None]:
        """Update full-training-set nnPU before each tree (beta=0, gamma=1)."""
        params = params.copy()
        iteration_names = [
            key for key in ("iterations", "n_estimators", "num_trees") if key in params
        ]
        if len(iteration_names) > 1:
            raise ValueError("Specify only one CatBoost iteration parameter")
        iterations = (
            int(cast(int, params.pop(iteration_names[0]))) if iteration_names else 1000
        )
        if iterations < 1:
            raise ValueError("iterations must be positive")
        stopping_rounds = params.pop("early_stopping_rounds", None)
        if stopping_rounds is not None and int(cast(int, stopping_rounds)) < 1:
            raise ValueError("early_stopping_rounds must be positive")
        use_best_model = bool(params.pop("use_best_model", validation is not None))
        params["iterations"] = 1
        # CatBoost's automatic learning rate would otherwise use a 1-tree run.
        params.setdefault("learning_rate", 0.03)
        params.pop("verbose", None)
        params["logging_level"] = "Silent"
        validation_data = (
            None if validation is None else validation.to_catboost(baseline=baseline)
        )
        if validation is not None:
            _pu_risks(
                np.full(len(validation.y), baseline),
                validation.y.to_numpy(),
                validation.weight.to_numpy(),
                prior,
            )
        train_labels = train.y.to_numpy()
        train_weights = train.weight.to_numpy()
        model = None
        best_model = None
        best_iteration = None
        best_risk = float("inf")

        for iteration in range(iterations):
            # CatBoost's derivative callback sees only a subset, so compute
            # this decision from the complete training pool between trees.
            logits = (
                np.full(len(train_labels), baseline)
                if model is None
                else np.asarray(
                    model.predict(train_data, prediction_type="RawFormulaVal"),
                    dtype=float,
                ).reshape(-1)
            )
            _, negative_risk = _pu_risks(logits, train_labels, train_weights, prior)
            objective.reverse_negative_risk = negative_risk < 0.0
            tree_params = {**params, "loss_function": objective}
            next_model = catboost.CatBoostClassifier(**tree_params)
            next_model.fit(train_data, eval_set=validation_data, init_model=model)
            model = next_model
            if iteration == 0 or (iteration + 1) % 50 == 0:
                logger.debug(
                    "nnPU tree %d/%d; training negative risk %.6g; correction=%s",
                    iteration + 1,
                    iterations,
                    negative_risk,
                    objective.reverse_negative_risk,
                )

            if validation_data is not None:
                assert validation is not None
                validation_logits = np.asarray(
                    model.predict(validation_data, prediction_type="RawFormulaVal"),
                    dtype=float,
                ).reshape(-1)
                positive_risk, negative_risk = _pu_risks(
                    validation_logits,
                    validation.y.to_numpy(),
                    validation.weight.to_numpy(),
                    prior,
                )
                risk = positive_risk + max(0.0, negative_risk)
                if risk < best_risk:
                    best_risk = risk
                    best_iteration = iteration
                    best_model = model
                if stopping_rounds is not None and iteration - cast(
                    int, best_iteration
                ) >= int(cast(int, stopping_rounds)):
                    break

        assert model is not None
        if validation is not None and use_best_model:
            assert best_model is not None
            model = best_model
        return model, best_iteration
