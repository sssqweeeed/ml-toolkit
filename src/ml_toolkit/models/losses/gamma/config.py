"""Training-only validation and construction of the Gamma backend parameters."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ....types import ModelParams

if TYPE_CHECKING:
    from ....data import Pool


def validate_gamma_data(data: Pool, name: str) -> None:
    targets, weights = data.y.to_numpy(), data.weight.to_numpy()
    # CatBoost stores both arrays as float32, even for CPU training. Validate
    # both representations so underflow cannot turn a positive label into zero.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        for values in (targets, targets.astype(np.float32)):
            if not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError(
                    f"Gamma {name} targets must be finite and strictly positive, "
                    "including after conversion to float32"
                )
        for values in (weights, weights.astype(np.float32)):
            total = np.sum(values, dtype=np.float64)
            if (
                not np.isfinite(values).all()
                or (values < 0).any()
                or not np.isfinite(total)
                or total <= 0
            ):
                raise ValueError(
                    f"Gamma {name} weights must be finite and non-negative with "
                    "positive total weight, including after conversion to float32"
                )


def configure_gamma(
    params: ModelParams, train: Pool, validation: Pool | None, *, log_link: bool
) -> None:
    validate_gamma_data(train, "training")
    if validation is not None:
        validate_gamma_data(validation, "validation")
    if not log_link and params.get("boost_from_average"):
        raise ValueError(
            "Gamma boost_from_average=True requires CatBoostLogLinkRegressionModel"
        )
    task_type = str(params.get("task_type", "CPU")).upper()
    if task_type not in ("CPU", "GPU"):
        raise ValueError("Gamma task_type must be 'CPU' or 'GPU'")
    params["task_type"] = task_type
    params.setdefault("boosting_type", "Plain")
    params.setdefault("leaf_estimation_method", "Newton")
    params.setdefault("leaf_estimation_iterations", 1)

    from .cpu import GammaMetric, GammaObjective

    metric = params.get("eval_metric", "Gamma")
    if isinstance(metric, str) and metric == "Gamma":
        params["eval_metric"] = GammaMetric()
    if task_type == "GPU":
        device = str(params.setdefault("devices", "0"))
        if not device.isdecimal():
            raise ValueError("Gamma GPU training requires one device, e.g. devices='0'")
        params["devices"] = device
        from .cuda import make_gamma_gpu_objective

        params["loss_function"] = make_gamma_gpu_objective()
    else:
        params["loss_function"] = GammaObjective()
