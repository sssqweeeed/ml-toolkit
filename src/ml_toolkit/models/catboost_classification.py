from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
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

from .._console import Logger
from ..data import validate_training_categories
from ..tracking.state import ClassificationState, read_state
from ..types import FloatArray, IntArray, ModelParams
from ._catboost import (
    feature_importances,
    native_fit,
    normalize_loss,
    prepare_frame,
    refit_parameters,
)
from .losses.pu.common import (
    UPUMetric as UPUMetric,
    NNPUMetric as NNPUMetric,
    PULogisticObjective,
    pu_risks,
)

if TYPE_CHECKING:
    from ..config import TrackingConfig
    from ..training.contracts import Pool

logger = Logger("ml_toolkit.models.catboost_classification")


class CatBoostClassificationModel(PythonModel):
    """Train and serve a binary CatBoost classifier with weighted log-odds.

    Parameters
    ----------
    params : dict or None, default None
        Copied CatBoost parameters; loss accepts Logloss or CrossEntropy.
        ``decision_threshold`` fixes the label cutoff and disables F1 tuning.
        Otherwise binary validation labels tune weighted F1. Without validation,
        or with soft CrossEntropy validation labels in [0, 1], the cutoff is 0.5.
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
        self._retained_iterations: int | None = None

    @property
    def n_iterations(self) -> int:
        """Return the number of fitted trees used for probability prediction."""
        return int(self.native_model.tree_count_)

    def shrink(self, n_iterations: int) -> Self:
        """Return an independent prefix, preserving baseline and threshold.

        n_iterations must be an integer from one through the fitted size.
        Shrinking never reselects a decision threshold. Validation F1 is cleared
        when predictions change; the copy's refit budget is its retained size.
        """
        from ._shrink import validate_size

        validate_size(n_iterations, self.n_iterations)
        native = self.native_model.copy()
        native.shrink(n_iterations, ntree_start=0)
        result = deepcopy(self, {id(self.native_model): native})
        result._retained_iterations = n_iterations
        if n_iterations < self.n_iterations:
            result._validation_f1 = None
        return result

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
            Supervised F1-selected cutoff, 0.5 without binary validation labels,
            or explicitly supplied decision_threshold.

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
        """Return the weighted positive-class F1 used for threshold tuning, if available.

        Notes
        -----
        None for PU models, supervised fits without validation or with soft
        validation labels, and fits with an explicit decision_threshold.
        """
        return self._validation_f1

    @property
    def best_iteration(self) -> int | None:
        """Return the zero-based best iteration index, or None when unavailable."""
        value = self._best_iteration
        if value is None:
            value = self.native_model.get_best_iteration()
        if value is None or value < 0 or value >= self.n_iterations:
            return None
        return int(value)

    def feature_importances(self, data: Pool) -> dict[str, float]:
        """Rank the current fitted feature columns.

        Parameters
        ----------
        data : Pool
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
            The best iteration selected on validation plus one, when available;
            otherwise the original iteration budget. Early stopping is removed
            and use_best_model is disabled. Learned baselines are recalculated.
            Includes the selected decision threshold when one exists.

        Raises
        ------
        RuntimeError
            The adapter has not been fitted.
        """
        self.native_model
        iteration = (
            self.best_iteration
            if self._retained_iterations is None
            else self._retained_iterations - 1
        )
        params = refit_parameters(self.params, iteration)
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
            catboost.CatBoostClassifier,
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

    def log(
        self,
        sample: Pool,
        *,
        config: TrackingConfig | None = None,
        extra_artifacts: Mapping[str, str] | None = None,
    ) -> ModelInfo:
        """Package and log the fitted model with a serving input example.

        Parameters
        ----------
        sample : Pool
            Serving example rows, capped by config.input_example_rows.
        config : TrackingConfig or None, default None
            Artifact name, positive example-row limit, registry name, and serving
            environment. None uses TrackingConfig defaults.

        extra_artifacts : Mapping[str, str] or None, default None
            Additional files or directories bundled with the model. Names must
            not conflict with the adapter's own artifacts.

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
        sample_frame = sample.to_model_input(
            n_rows=config.input_example_rows, categorical_as_strings=True
        ).loc[:, self.feature_names]
        model_info = self._log_artifacts(
            sample_frame,
            self.predict_frame(sample_frame),
            name=name,
            registered_model_name=registered_model_name,
            conda_env=conda_env,
            extra_artifacts=extra_artifacts,
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
        y_true: FloatArray, probabilities: FloatArray, weight: FloatArray | None = None
    ) -> tuple[float, float]:
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
        extra_artifacts: Mapping[str, str] | None = None,
    ) -> ModelInfo:
        from ..tracking.model_artifacts import log_catboost

        return log_catboost(
            self,
            sample_frame,
            sample_prediction,
            name=name,
            registered_model_name=registered_model_name,
            conda_env=conda_env,
            extra_artifacts=extra_artifacts,
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
            "retained_iterations": self._retained_iterations,
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
        self._retained_iterations = state.get("retained_iterations")
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
            pu_risks(
                np.full(len(validation), baseline),
                validation.y.to_numpy(),
                validation.weight.to_numpy(),
                self._pu_prior,
            )
        if self._pu_loss == "nnPU":
            objective = PULogisticObjective(
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
            from .losses.pu.cuda import make_upu_gpu_objective

            params["loss_function"] = make_upu_gpu_objective(
                positive_scale, unlabeled_scale
            )
        else:
            from .losses.pu.cpu import UPUObjective

            params["loss_function"] = UPUObjective(positive_scale, unlabeled_scale)
        if validation is not None:
            params.setdefault("use_best_model", True)
        return super()._fit_backend(train, validation, baseline, params)

    def _configure_threshold(
        self, validation: Pool | None, threshold: float | None
    ) -> None:
        self._threshold = threshold

    @staticmethod
    def _fit_nnpu(
        *,
        train: Pool,
        validation: Pool | None,
        train_data: catboost.Pool,
        baseline: float,
        prior: float,
        objective: PULogisticObjective,
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
            pu_risks(
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
            _, negative_risk = pu_risks(logits, train_labels, train_weights, prior)
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
                positive_risk, negative_risk = pu_risks(
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
