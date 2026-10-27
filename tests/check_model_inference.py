"""Check bundled model inference in a process without training dependencies."""

import importlib.abc
import importlib.util
import sys
from collections.abc import Sequence
from importlib.machinery import ModuleSpec
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import ModuleType


for package in ("optuna", "numba", "ml_toolkit"):
    assert importlib.util.find_spec(package) is None, package
    try:
        version(package)
    except PackageNotFoundError:
        pass
    else:
        raise AssertionError(f"Unexpected installed package: {package}")


class BlockTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        if fullname.split(".")[0] in ("optuna", "numba") or fullname.startswith(
            ("ml_toolkit.training", "ml_toolkit.metrics")
        ):
            raise ImportError(f"Inference imported a training dependency: {fullname}")
        return None


sys.meta_path.insert(0, BlockTrainingImports())

import mlflow
import numpy as np
import pandas as pd

root = Path(sys.argv[1])
model_directory = Path(sys.argv[3])
mlflow.set_tracking_uri(sys.argv[2])
mlflow.set_experiment("toolkit-test")
loaded = mlflow.pyfunc.load_model(str(model_directory))
model = loaded.unwrap_python_model()
assert type(model).__name__ == sys.argv[4]
if sys.argv[5]:
    assert type(model.model).__name__ == sys.argv[5]

import ml_toolkit

assert Path(ml_toolkit.__file__).is_relative_to(model_directory / "code")
frame = pd.read_pickle(root / "frame.pkl")
expected = np.load(root / "prediction.npy")
prediction = loaded.predict(frame)
assert prediction.shape == expected.shape
np.testing.assert_allclose(prediction, expected)
np.testing.assert_allclose(loaded.predict(frame.iloc[:1]), expected[:1])
if sys.argv[4] != "LightGBMServingModel":
    np.testing.assert_allclose(model.predict_frame(frame), expected)
    from ml_toolkit.tracking.model_artifacts import log_catboost

    with mlflow.start_run():
        info = log_catboost(model, frame.head(5), expected[:5], name="relogged")
    again = mlflow.pyfunc.load_model(info.model_uri)
    np.testing.assert_allclose(again.predict(frame), expected)
    assert type(again.unwrap_python_model()) is type(model)

assert not any(
    name.split(".")[0] in ("optuna", "numba")
    or name.startswith(("ml_toolkit.training", "ml_toolkit.metrics"))
    for name in sys.modules
)
