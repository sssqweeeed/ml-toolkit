"""Small deterministic data and model doubles shared by workflow tests."""

from dataclasses import dataclass
from datetime import date, timedelta

import mlflow
import numpy as np
import polars as pl

from examples.project_pool import ProjectPool
from ml_toolkit.metrics import MetricScores


@dataclass(frozen=True)
class ProjectData:
    X: pl.DataFrame
    y: pl.Series
    time: pl.Series
    weight: pl.Series
    cat_features: tuple[str, ...] = ()
    numerical_features: tuple[str, ...] = ()
    object_id: pl.Series | None = None
    business_type: pl.Series | None = None

    def __len__(self):
        return len(self.X)

    def take(self, indices):
        index = np.asarray(indices, dtype=np.int64)
        return ProjectData(
            self.X[index],
            self.y.gather(index),
            self.time.gather(index),
            self.weight.gather(index),
            self.cat_features,
        )

    def select_features(self, names):
        return ProjectData(
            self.X.select(names),
            self.y,
            self.time,
            self.weight,
            tuple(name for name in self.cat_features if name in names),
        )

    def with_target(self, target):
        values = (
            target if isinstance(target, pl.Series) else pl.Series("target", target)
        )
        return ProjectData(self.X, values, self.time, self.weight, self.cat_features)

    def to_catboost(self, baseline=None):
        import catboost

        baseline_values = None if baseline is None else np.full(len(self), baseline)
        return catboost.Pool(
            self.X.to_pandas(),
            label=self.y.to_numpy(),
            weight=self.weight.to_numpy(),
            cat_features=list(self.cat_features),
            baseline=baseline_values,
        )

    def to_model_input(self, n_rows=None, *, categorical_as_strings=False):
        data = ProjectPool(
            self.X, self.y, self.time, self.weight, cat_features=self.cat_features
        )
        return data.to_model_input(
            n_rows, categorical_as_strings=categorical_as_strings
        )


class MeanModel:
    def __init__(self, params):
        self.params = params
        self.mean = None

    @property
    def best_iteration(self):
        return None

    def fit(self, train, validation=None):
        from sklearn.base import BaseEstimator

        class Native(BaseEstimator):
            def fit(self, X):
                self.feature_names_in_ = np.asarray(X.columns)
                self.feature_importances_ = np.arange(
                    len(X.columns), 0, -1, dtype=float
                )
                return self

        self._native = Native().fit(train.X)
        self.mean = float(
            np.average(train.y.to_numpy(), weights=train.weight.to_numpy())
        )
        return self

    def predict_pool(self, data):
        return np.full(len(data), self.mean)

    @property
    def native_model(self):
        return self._native

    def feature_importances(self, data):
        return dict(
            zip(
                self._native.feature_names_in_,
                self._native.feature_importances_,
                strict=True,
            )
        )

    def get_refit_params(self):
        params = self.params.copy()
        if self.best_iteration is not None:
            key = next(
                (
                    name
                    for name in ("iterations", "n_estimators", "num_trees")
                    if name in params
                ),
                "iterations",
            )
            params[key] = self.best_iteration + 1
        params.pop("early_stopping_rounds", None)
        if "use_best_model" in params:
            params["use_best_model"] = False
        return params

    def log(self, sample, *, config=None):
        mlflow.log_dict({"mean": self.mean}, "model.json")


def mae(y, prediction, weight):
    return MetricScores({"mae": np.average(np.abs(y - prediction), weights=weight)})


def make_data():
    length = 24
    return ProjectData(
        X=pl.DataFrame(
            {
                "x": np.arange(length, dtype=float),
                "unused": np.arange(length, dtype=float) % 3,
            }
        ),
        y=pl.Series("target", np.arange(length, dtype=float) + 1),
        time=pl.Series(
            "date", [date(2024, 1, 1) + timedelta(days=30 * n) for n in range(length)]
        ),
        weight=pl.Series("weight", np.ones(length)),
    )


def setup_mlflow(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment_id = mlflow.create_experiment(
        "toolkit-test", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    mlflow.set_experiment(experiment_id=experiment_id)


def make_identified_data():
    source = make_data()
    return ProjectPool(
        source.X,
        source.y,
        source.time,
        source.weight,
        object_id=pl.Series("id", np.arange(len(source))),
        business_type=pl.Series("group", ["a", "b"] * (len(source) // 2)),
        numerical_features=tuple(source.X.columns),
    )


def list_all_artifacts(client, run_id, path=""):
    result = []
    for item in client.list_artifacts(run_id, path):
        result.extend(
            list_all_artifacts(client, run_id, item.path)
            if item.is_dir
            else [item.path]
        )
    return result
