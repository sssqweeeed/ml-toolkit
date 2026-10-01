from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Self

import catboost
import mlflow
import numpy as np
import pandas as pd
import polars as pl
from mlflow.pyfunc.utils import pyfunc

if TYPE_CHECKING:
    from ..training.contracts import TrainingData

logger = logging.getLogger(__name__)


def _pu_risks(
    logits: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
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

    positive_risk = prior * np.average(
        np.logaddexp(0.0, -logits[positive]), weights=weights[positive]
    )
    negative_risk = np.average(
        np.logaddexp(0.0, logits[unlabeled]), weights=weights[unlabeled]
    ) - prior * np.average(
        np.logaddexp(0.0, logits[positive]), weights=weights[positive]
    )
    return float(positive_risk), float(negative_risk)


class _PURiskMetric:
    """Evaluate PU risk over the complete CatBoost training or eval pool."""

    def __init__(self, prior: float, *, non_negative: bool) -> None:
        self.prior = prior
        self.non_negative = non_negative

    def is_max_optimal(self) -> bool:
        return False

    def get_final_error(self, error: float, weight: float) -> float:
        return error / weight

    def evaluate(self, approxes, target, weight) -> tuple[float, float]:
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
    def __init__(self, prior: float) -> None:
        super().__init__(prior, non_negative=False)


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

    def calc_ders_range(self, approxes, targets, weights):
        result = []
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


class CatBoostModel(mlflow.pyfunc.PythonModel):  # type: ignore
    """Classifier for labels 0/1 with a log-odds baseline and F1 threshold."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = {} if params is None else params.copy()
        self._model: catboost.CatBoostClassifier | None = None
        self._baseline: float | None = None
        self._threshold: float | None = None
        self._validation_f1: float | None = None
        self._pu_loss: str | None = None
        self._pu_prior: float | None = None
        self._best_iteration: int | None = None

    def load_context(self, context) -> None:
        self._model = catboost.CatBoostClassifier()
        self._model.load_model(context.artifacts["model"])
        self.params = self._model.get_all_params()
        with open(context.artifacts["config"], encoding="utf-8") as handle:
            config = json.load(handle)
        self._baseline = float(config["baseline"])
        # Older artifacts have no tuned threshold.
        self._threshold = float(config.get("threshold", 0.5))
        score = config.get("validation_f1")
        self._validation_f1 = None if score is None else float(score)
        self._pu_loss = config.get("pu_loss")
        prior = config.get("pu_prior")
        self._pu_prior = None if prior is None else float(prior)
        self._best_iteration = config.get("best_iteration")
        if self._pu_loss is not None:
            self.params = config.get("training_params", self.params)

    @property
    def native_model(self) -> catboost.CatBoostClassifier:
        if self._model is None:
            raise RuntimeError("Model has not been fitted")
        return self._model

    @property
    def baseline(self) -> float:
        if self._baseline is None:
            raise RuntimeError("Model has not been fitted")
        return self._baseline

    @property
    def threshold(self) -> float:
        if self._threshold is None:
            raise RuntimeError("Model has not been fitted")
        return self._threshold

    @property
    def validation_f1(self) -> float | None:
        """Unweighted positive-class F1 used for threshold tuning, if available."""
        return self._validation_f1

    @property
    def best_iteration(self) -> int | None:
        if self._pu_loss == "nnPU":
            return self._best_iteration
        value = self.native_model.get_best_iteration()
        if value is None or value < 0:
            return None
        return int(value)

    @property
    def feature_names(self) -> list[str]:
        names = self.native_model.feature_names_
        if not names:
            raise RuntimeError("Fitted model does not contain feature names")
        return list(names)

    def fit(self, train: TrainingData, validation: TrainingData | None = None) -> Self:
        params = self.params.copy()
        requested_loss = params.pop("loss", None)
        native_loss = params.get("loss_function")
        if (
            requested_loss is not None
            and native_loss is not None
            and requested_loss not in ("uPU", "nnPU")
            and requested_loss != native_loss
        ):
            raise ValueError("Specify only one of loss and loss_function")
        selected_loss = requested_loss if requested_loss is not None else native_loss
        if selected_loss not in ("uPU", "nnPU"):
            if requested_loss is not None:
                params["loss_function"] = requested_loss
            if "pu_prior" in params:
                raise ValueError("pu_prior requires loss='uPU' or loss='nnPU'")
            baseline = self._calculate_baseline(train)
            pu_prior = None
            pu_objective = None
            self._pu_loss = None
        else:
            params.pop("loss_function", None)
            prior_value = params.pop("pu_prior", None)
            if prior_value is None:
                raise ValueError("PU loss requires pu_prior in params")
            pu_prior = float(prior_value)
            if not 0 < pu_prior < 1 or not np.isfinite(pu_prior):
                raise ValueError("pu_prior must be finite and strictly between 0 and 1")
            if str(params.get("task_type", "CPU")).upper() != "CPU":
                raise ValueError("Python PU objectives require task_type='CPU'")
            for name in ("class_weights", "auto_class_weights", "scale_pos_weight"):
                if name in params:
                    raise ValueError(f"{name} is incompatible with PU loss")
            labels = train.y.to_numpy()
            weights = train.weight.to_numpy()
            baseline = float(math.log(pu_prior / (1.0 - pu_prior)))
            _pu_risks(np.full(len(labels), baseline), labels, weights, pu_prior)
            positive_weight = float(weights[labels == 1].sum())
            unlabeled_weight = float(weights[labels == 0].sum())
            total_weight = positive_weight + unlabeled_weight
            pu_objective = _PULogisticObjective(
                prior=pu_prior,
                positive_scale=pu_prior * total_weight / positive_weight,
                unlabeled_scale=total_weight / unlabeled_weight,
            )
            params["leaf_estimation_method"] = "Gradient"
            params["leaf_estimation_iterations"] = 1
            params["boosting_type"] = "Plain"
            eval_metric = params.get("eval_metric", selected_loss)
            if eval_metric in ("uPU", "nnPU"):
                metric_class = UPUMetric if eval_metric == "uPU" else NNPUMetric
                params["eval_metric"] = metric_class(pu_prior)
            self._pu_loss = selected_loss
        params["boost_from_average"] = False
        train_data = train.to_catboost(baseline=baseline)

        if selected_loss == "nnPU":
            assert pu_objective is not None and pu_prior is not None
            model, self._best_iteration = self._fit_nnpu(
                train=train,
                validation=validation,
                train_data=train_data,
                baseline=baseline,
                prior=pu_prior,
                objective=pu_objective,
                params=params,
            )
        else:
            self._best_iteration = None
            if validation is None:
                params.pop("early_stopping_rounds", None)
                params["use_best_model"] = False
            if pu_objective is not None:
                params["loss_function"] = pu_objective
                if validation is not None:
                    params.setdefault("use_best_model", True)
            model = catboost.CatBoostClassifier(**params)
            if validation is None:
                model.fit(train_data)
            else:
                model.fit(
                    train_data,
                    eval_set=validation.to_catboost(baseline=baseline),
                )

        self._model = model
        self._baseline = baseline
        self._pu_prior = pu_prior
        # Reset on every fit, including refits without validation.
        self._threshold = 0.5
        self._validation_f1 = None
        if validation is not None:
            self._threshold, self._validation_f1 = self._select_threshold(
                validation.y.to_numpy(),
                self.predict_pool(validation),
            )
        return self

    @staticmethod
    def _fit_nnpu(
        *,
        train: TrainingData,
        validation: TrainingData | None,
        train_data: catboost.Pool,
        baseline: float,
        prior: float,
        objective: _PULogisticObjective,
        params: dict[str, Any],
    ) -> tuple[catboost.CatBoostClassifier, int | None]:
        """Update full-training-set nnPU before each tree (beta=0, gamma=1)."""
        params = params.copy()
        iteration_names = [
            key for key in ("iterations", "n_estimators", "num_trees") if key in params
        ]
        if len(iteration_names) > 1:
            raise ValueError("Specify only one CatBoost iteration parameter")
        iterations = int(params.pop(iteration_names[0])) if iteration_names else 1000
        if iterations < 1:
            raise ValueError("iterations must be positive")
        stopping_rounds = params.pop("early_stopping_rounds", None)
        if stopping_rounds is not None and int(stopping_rounds) < 1:
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
                logger.info(
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
                if (
                    stopping_rounds is not None
                    and iteration - best_iteration >= int(stopping_rounds)
                ):
                    break

        assert model is not None
        if validation is not None and use_best_model:
            assert best_model is not None
            model = best_model
        return model, best_iteration

    def predict_pool(self, data: TrainingData) -> np.ndarray:
        return self.predict_frame(data.X)

    def predict_labels(self, data: pl.DataFrame | pd.DataFrame) -> np.ndarray:
        """Predict labels using probability >= the stored threshold."""
        return (self.predict_frame(data) >= self.threshold).astype(np.int64)

    def predict_labels_pool(self, data: TrainingData) -> np.ndarray:
        return self.predict_labels(data.X)

    def select_features(
        self,
        train: TrainingData,
        validation: TrainingData,
        num_features_to_select: int,
    ) -> list[str]:
        if self.params.get("loss", self.params.get("loss_function")) in ("uPU", "nnPU"):
            raise NotImplementedError(
                "CatBoost's built-in feature selection cannot optimize PU risk"
            )
        feature_names = train.X.columns
        if not 0 < num_features_to_select <= len(feature_names):
            raise ValueError(
                "num_features_to_select must be between 1 and the feature count"
            )
        if num_features_to_select == len(feature_names):
            return feature_names

        baseline = self._calculate_baseline(train)
        params = self.params.copy()
        params["boost_from_average"] = False
        model = catboost.CatBoostClassifier(**params)
        summary = model.select_features(
            train.to_catboost(baseline=baseline),
            eval_set=validation.to_catboost(baseline=baseline),
            features_for_select=feature_names,
            num_features_to_select=num_features_to_select,
            steps=min(20, len(feature_names) - num_features_to_select),
            algorithm=catboost.EFeaturesSelectionAlgorithm.RecursiveByShapValues,
            shap_calc_type=catboost.EShapCalcType.Regular,
            train_final_model=False,
            plot=False,
            logging_level="Verbose",
        )
        return list(summary["selected_features_names"])

    @pyfunc
    def predict(self, context, model_input: pd.DataFrame, params=None) -> np.ndarray:  # type: ignore
        return self.predict_frame(model_input)

    @property
    def cat_features(self) -> list[str]:
        return [
            self.feature_names[index]
            for index in self.native_model.get_cat_feature_indices()
        ]

    @property
    def numerical_features(self) -> list[str]:
        categorical = set(self.cat_features)
        return [name for name in self.feature_names if name not in categorical]

    def log(
        self,
        sample: TrainingData,
        *,
        name="model",
        registered_model_name=None,
        conda_env=None,
    ) -> Any:
        sample_frame = sample.to_model_input(n_rows=5).loc[:, self.feature_names]
        model_info = self._log_artifacts(
            sample_frame,
            self.predict_frame(sample_frame),
            name=name,
            registered_model_name=registered_model_name,
            conda_env=conda_env,
        )
        mlflow.log_param("baseline", self.baseline)
        mlflow.log_param("threshold", self.threshold)
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

    def _log_artifacts(
        self,
        sample_frame,
        sample_prediction,
        *,
        name,
        registered_model_name=None,
        conda_env=None,
        metadata=None,
        code_paths=None,
        source=None,
    ) -> Any:
        from .artifacts import log_model_from_code

        metadata = {
            **(metadata or {}),
            "numerical_features": self.numerical_features,
            "cat_features": self.cat_features,
            "baseline": self.baseline,
            "threshold": self.threshold,
            "validation_f1": self.validation_f1,
            "pu_loss": self._pu_loss,
            "pu_prior": self._pu_prior,
            "best_iteration": self._best_iteration,
            **({"training_params": self.params} if self._pu_loss else {}),
        }
        with TemporaryDirectory(prefix="catboost-mlflow-") as directory:
            model_path = Path(directory) / "model.cbm"
            config_path = Path(directory) / "config.json"
            self.native_model.save_model(str(model_path))
            config_path.write_text(json.dumps(metadata), encoding="utf-8")
            return log_model_from_code(
                source=Path(__file__).resolve() if source is None else source,
                sample_frame=sample_frame,
                sample_prediction=sample_prediction,
                artifacts={"model": str(model_path), "config": str(config_path)},
                metadata=metadata,
                params=self.native_model.get_all_params(),
                name=name,
                registered_model_name=registered_model_name,
                conda_env=conda_env,
                code_paths=code_paths,
                packages=("catboost", "polars", "pyarrow"),
            )

    def predict_frame(self, data: pl.DataFrame | pd.DataFrame) -> np.ndarray:
        if not isinstance(data, (pl.DataFrame, pd.DataFrame)):
            raise TypeError("Prediction data must be a Polars or Pandas DataFrame")

        feature_names = self.feature_names
        missing = sorted(set(feature_names) - set(data.columns))
        if missing:
            raise ValueError(f"Missing model input columns: {missing}")

        if isinstance(data, pl.DataFrame):
            feature_frame = data.select(feature_names).to_pandas()
        else:
            feature_frame = data.loc[:, feature_names]

        raw_predictions = np.asarray(
            self.native_model.predict(
                feature_frame,
                prediction_type="RawFormulaVal",
            ),
            dtype=float,
        ).reshape(-1)

        # DataFrame predictions have no attached baseline; add it exactly once.
        logits = raw_predictions + self.baseline

        # Numerically stable sigmoid, without an additional dependency.
        return np.exp(-np.logaddexp(0.0, -logits))

    @staticmethod
    def _select_threshold(
        y_true: np.ndarray, probabilities: np.ndarray
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
        return float(threshold), float(scores[best_index])

    @staticmethod
    def _calculate_baseline(data: TrainingData) -> float:
        """Calculate the baseline log-odds from the positive class prevalence."""
        weights = data.weight.to_numpy()
        y_values = data.y.to_numpy()
        weighted_sum = np.average(y_values, weights=weights)
        if not np.isfinite(weighted_sum):
            raise ValueError("Weighted target mean must be finite")
        epsilon = 1e-7
        weighted_sum = np.clip(weighted_sum, epsilon, 1 - epsilon)
        return float(np.log(weighted_sum / (1 - weighted_sum)))


mlflow.models.set_model(CatBoostModel())  # type: ignore
