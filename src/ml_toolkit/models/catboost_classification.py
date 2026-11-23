from __future__ import annotations

from typing import Any, Self, cast

import catboost
import mlflow
import numpy as np
import pandas as pd
import polars as pl
from mlflow.pyfunc.utils import pyfunc

from .._console import Logger
from ..data import Pool, validate_training_categories
from ..types import FloatArray, IntArray, ModelParams
from ._catboost import CatBoostAdapter, native_fit, normalize_loss
from .state import ClassificationState

logger = Logger("ml_toolkit.models.catboost_classification")


class CatBoostClassificationModel(
    CatBoostAdapter[catboost.CatBoostClassifier, ClassificationState]
):
    """Train and serve a binary CatBoost classifier with weighted log-odds.

    Parameters
    ----------
    params : dict or None, default None
        Copied CatBoost parameters; loss accepts Logloss or CrossEntropy.
        ``decision_threshold`` fixes the label cutoff and disables F1 tuning.
        Otherwise binary validation labels tune weighted F1. Without validation,
        or with soft CrossEntropy validation labels in [0, 1], the cutoff is 0.5.
    """

    _native_type = catboost.CatBoostClassifier
    baseline_mode = "raw_prediction"

    def __init__(self, params: ModelParams | None = None) -> None:
        super().__init__(params)
        self._threshold: float | None = None
        self._validation_f1: float | None = None
        self._pu_loss: str | None = None
        self._pu_prior: float | None = None

    def shrink(self, n_iterations: int) -> Self:
        """Return an independent prefix, preserving baseline and threshold.

        n_iterations must be an integer from one through the fitted size.
        Shrinking never reselects a decision threshold. Validation F1 is cleared
        when predictions change; the copy's refit budget is its retained size.
        """
        result = super().shrink(n_iterations)
        if n_iterations < self.n_iterations:
            result._validation_f1 = None
        return result

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
            Supervised F1-selected cutoff, 0.5 without binary validation labels,
            or explicitly supplied decision_threshold.

        Raises
        ------
        RuntimeError
            The model is unfitted or a PU model has no explicit decision threshold.
        """
        self._require_fitted()
        if self._threshold is None:
            raise RuntimeError(
                "PU label prediction requires an explicit decision_threshold"
            )
        return self._threshold

    @property
    def validation_f1(self) -> float | None:
        """Return the weighted positive-class F1 used for threshold tuning, if available.

        Notes
        -----
        None for PU models, supervised fits without validation or with soft
        validation labels, and fits with an explicit decision_threshold.
        """
        return self._validation_f1

    def get_refit_params(self) -> ModelParams:
        """Return copied constructor parameters for final fitting.

        Includes the selected decision threshold when one exists; the iteration
        budget follows ``CatBoostRegressionModel.get_refit_params``.
        """
        params = super().get_refit_params()
        if self._threshold is not None:
            params["decision_threshold"] = self._threshold
        return params

    @property
    def task(self) -> str:
        """Return the supervised binary classification task identifier."""
        return "classification"

    def fit(self, train: Pool, validation: Pool | None = None) -> Self:
        """Fit the classifier with the subclass's baseline and training policy.

        Parameters
        ----------
        train : Pool
            Raw categorical values must not stringify to the reserved "unknown".
            Aligned features, labels and exposure weights.
        validation : Pool or None, default None
            Subject to the same reserved-category check as training.
            Optional holdout for early stopping. Binary labels also enable
            supervised F1 tuning unless decision_threshold is explicit;
            soft CrossEntropy labels retain the default cutoff of 0.5.

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
        validate_training_categories(train)
        if validation is not None:
            validate_training_categories(validation)
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
        self._retained_iterations = None
        self._validate_params(params, train, validation)
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

    def _validate_params(
        self, params: ModelParams, train: Pool, validation: Pool | None
    ) -> None:
        loss = params.setdefault("loss_function", "Logloss")
        if not isinstance(loss, str) or loss.split(":", 1)[0] not in (
            "Logloss",
            "CrossEntropy",
        ):
            raise ValueError("Binary classification requires Logloss or CrossEntropy")
        if "pu_prior" in params:
            raise ValueError("pu_prior requires CatBoostPUClassificationModel")
        for name, data in (("targets", train), ("validation targets", validation)):
            if data is None:
                continue
            y = data.y.to_numpy()
            if not np.isfinite(y).all() or (y < 0).any() or (y > 1).any():
                raise ValueError(
                    f"Binary classification {name} must be between 0 and 1"
                )
            if loss.split(":", 1)[0] == "Logloss" and not np.isin(y, [0, 1]).all():
                raise ValueError(f"Logloss {name} must be binary")

    def _fit_backend(
        self,
        train: Pool,
        validation: Pool | None,
        baseline: float,
        params: ModelParams,
    ) -> catboost.CatBoostClassifier:
        return native_fit(
            self._native_type,
            params,
            train.to_catboost(baseline=baseline),
            None if validation is None else validation.to_catboost(baseline=baseline),
        )

    def _configure_threshold(
        self, validation: Pool | None, threshold: float | None
    ) -> None:
        self._threshold = 0.5 if threshold is None else threshold
        if (
            validation is not None
            and threshold is None
            and np.isin(validation.y.to_numpy(), [0, 1]).all()
        ):
            self._threshold, self._validation_f1 = self._select_threshold(
                validation.y.to_numpy(),
                self.predict_pool(validation),
                validation.weight.to_numpy().astype(float),
            )

    def predict_pool(self, data: Pool) -> FloatArray:
        """Predict positive-class probabilities for project data.

        Parameters
        ----------
        data : Pool
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
        return self.predict_frame(data.X)

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

    def predict_labels_pool(self, data: Pool) -> IntArray:
        """Predict thresholded binary labels for project-owned data.

        Parameters
        ----------
        data : Pool
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
        return self.predict_labels(data.X)

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

    @staticmethod
    def _select_threshold(
        y_true: FloatArray, probabilities: FloatArray, weight: FloatArray | None = None
    ) -> tuple[float, float]:

        # Serving must not import ml_toolkit.metrics (checked by the inference test).
        from ..metrics.classification import binary_threshold_scores

        y = np.asarray(y_true, dtype=float)
        p = np.asarray(probabilities, dtype=float)
        w = np.ones(len(y)) if weight is None else np.asarray(weight, dtype=float)
        thresholds, _, _, scores = binary_threshold_scores(y, p, w)
        if not np.any((y == 1) & (w > 0)):
            return 0.5, 0.0
        best = int(np.argmax(scores))
        return float(thresholds[best]), float(scores[best])

    def _calculate_baseline(self, train: Pool) -> float:
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
            Categorical columns are formatted exactly as in training. Pandas
            integer categoricals with missing values must use the nullable
            ``Int64`` dtype; a ``float64`` column formats as ``"1.0"``.
            Object-dtype categorical columns are converted with pandas string
            formatting.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Positive-class probabilities. PU scores need not be calibrated.
        """
        logger.debug("Predicting with CatBoost classifier: %s rows", len(data))
        frame = self._input_frame(data)
        logits = (
            np.asarray(
                self._require_fitted().predict(frame, prediction_type="RawFormulaVal"),
                dtype=float,
            )
            + self.baseline
        )
        return np.exp(-np.logaddexp(0.0, -logits))

    def export_state(self) -> ClassificationState:
        return {
            "version": 1,
            "model_type": type(self).__name__,
            "prediction_type": "Probability",
            "baseline": self.baseline,
            "baseline_mode": self.baseline_mode,
            "threshold": self._threshold,
            "validation_f1": self._validation_f1,
            "pu_loss": self._pu_loss,
            "pu_prior": self._pu_prior,
            "best_iteration": self._best_iteration,
            "training_params": self.params.copy(),
            "retained_iterations": self._retained_iterations,
        }

    def _log_fitted_parameters(self) -> None:
        mlflow.log_param("baseline", self.baseline)
        if self._threshold is not None:
            mlflow.log_param("threshold", self._threshold)
        if self.validation_f1 is not None:
            mlflow.log_metric("validation_f1_at_threshold", self.validation_f1)

    def _restore_fitted_fields(self, state: ClassificationState) -> None:
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
        self._require_fitted()
        assert self._pu_loss is not None
        return self._pu_loss

    @property
    def pu_prior(self) -> float:
        """Return the fitted positive population prior."""
        self._require_fitted()
        assert self._pu_prior is not None
        return self._pu_prior

    def _validate_params(
        self, params: ModelParams, train: Pool, validation: Pool | None
    ) -> None:
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
        # Training-only compiled code stays out of serving environments.
        from .losses.pu import pu_risks

        pu_risks(
            np.zeros(len(train)),
            train.y.to_numpy(),
            train.weight.to_numpy(),
            prior,
        )

    def _calculate_baseline(self, train: Pool) -> float:
        assert self._pu_prior is not None
        return float(np.log(self._pu_prior / (1.0 - self._pu_prior)))

    def _fit_backend(
        self,
        train: Pool,
        validation: Pool | None,
        baseline: float,
        params: ModelParams,
    ) -> catboost.CatBoostClassifier:
        assert self._pu_prior is not None
        # Training-only compiled code stays out of serving environments.
        from .losses.pu import (
            NNPUMetric,
            PULogisticObjective,
            UPUMetric,
            UPUObjective,
            fit_nnpu,
            pu_risks,
        )

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
        if self._pu_loss == "nnPU":
            objective = PULogisticObjective(
                self._pu_prior, positive_scale, unlabeled_scale
            )
            model, self._best_iteration = fit_nnpu(
                train=train,
                validation=validation,
                baseline=baseline,
                prior=self._pu_prior,
                objective=objective,
                params=params,
            )
            return model
        if validation is not None:
            pu_risks(
                np.full(len(validation), baseline),
                validation.y.to_numpy(),
                validation.weight.to_numpy(),
                self._pu_prior,
            )
        if params["task_type"] == "GPU":
            from .losses.pu_cuda import make_upu_gpu_objective

            params["loss_function"] = make_upu_gpu_objective(
                positive_scale, unlabeled_scale
            )
        else:
            params["loss_function"] = UPUObjective(positive_scale, unlabeled_scale)
        if validation is not None:
            params.setdefault("use_best_model", True)
        return super()._fit_backend(train, validation, baseline, params)

    def _configure_threshold(
        self, validation: Pool | None, threshold: float | None
    ) -> None:
        self._threshold = threshold
