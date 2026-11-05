"""Scalar report metric validation without tracking side effects."""

import re

import numpy as np


def validate_report_metrics(values: dict[str, float]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in values.items():
        if (
            not re.fullmatch(r"[\w .\-/]+", key)
            or not key.strip()
            or any(part in ("", ".", "..") for part in key.split("/"))
        ):
            raise ValueError(f"Invalid report metric name: {key!r}")
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError(f"Report metric {key!r} must be finite")
        metrics[key] = numeric
    return metrics
