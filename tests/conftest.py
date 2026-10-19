"""Keep numerical workers, plots, and tracking state isolated during tests."""

import os
from collections.abc import Iterator

# Set before test collection imports numerical backends. Users may override these
# limits explicitly; xdist workers otherwise must not each consume every CPU.
for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "POLARS_MAX_THREADS",
):
    os.environ.setdefault(variable, "1")
os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt
import mlflow
import pytest


@pytest.fixture(autouse=True)
def clean_process_state() -> Iterator[None]:
    tracking_uri = mlflow.get_tracking_uri()
    yield
    while mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.set_tracking_uri(tracking_uri)
    plt.close("all")
