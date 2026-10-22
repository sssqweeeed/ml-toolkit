"""Shared real-data setup for the training, splitting and selection guides."""

import mlflow

from examples.gallery_data import load_bike_pool
from ml_toolkit.config import TrackingConfig, TrainingConfig
from ml_toolkit.data import Pool
from ml_toolkit.types import ModelParams


def prepare_training() -> tuple[Pool, ModelParams, TrainingConfig]:
    """Load cached Bike Sharing hours and configure a local MLflow experiment.

    Run ``python -m examples.prepare_data --dataset bike`` first. This function
    performs no downloads. It uses ``mlflow.db`` in the working directory and
    disables automatic model packaging; individual examples demonstrate packaging
    explicitly when needed.
    """
    data = load_bike_pool(download_if_missing=False)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("ml-toolkit-examples")
    params: ModelParams = {
        "iterations": 40,
        "depth": 4,
        "loss_function": "Poisson",
        "random_seed": 42,
        "thread_count": 2,
        "verbose": False,
        "allow_writing_files": False,
    }
    config = TrainingConfig(
        tracking=TrackingConfig(log_test_model=False, log_final_model=False)
    )
    return data, params, config
