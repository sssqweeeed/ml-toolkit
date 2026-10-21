"""Run Gamma's CUDA kernel in an isolated Numba CUDA simulator process."""

import inspect

import numpy as np
from numba import cuda

from ml_toolkit.models.losses.gamma.cpu import gamma_derivatives
from ml_toolkit.models.losses.gamma.cuda import make_gamma_gpu_objective

method = type(make_gamma_gpu_objective()).calc_ders_range_gpu
assert inspect.getsource(method).count("self") == 1
kernel = cuda.jit(method)
y = np.array([1e-20, 0.1, 1, 10, 1e20, 2], dtype=np.float32)
a = (np.log(y.astype(float)) + [-2, -0.1, 0, 0.2, 2, 0]).astype(np.float32)
for weighted in (False, True):
    weights = (
        np.array([1, 0.5, 2, 3, 1, 0], dtype=np.float32)
        if weighted
        else np.empty(0, dtype=np.float32)
    )
    raw = a.copy()
    if weighted:
        raw[-1] = -1000  # Must skip overflow before evaluating zero-weight rows.
    cpu = gamma_derivatives(raw.astype(float), y, weights if weighted else None)
    terms = np.zeros(len(y))
    active = weights > 0 if weighted else np.ones(len(y), dtype=bool)
    terms[active] = -(
        raw[active].astype(float)
        + y[active].astype(float) * np.exp(-raw[active].astype(float))
    )
    expected_value = np.sum(terms * (weights if weighted else 1))
    for output_value, output_derivatives in (
        (True, True),
        (True, False),
        (False, True),
    ):
        value = np.zeros(1 if output_value else 0, dtype=np.float32)
        d1 = np.full(len(y) if output_derivatives else 0, np.nan, dtype=np.float32)
        d2 = d1.copy()
        kernel[3, 4](0, raw, y, weights, value, d1, d2)
        if output_value:
            np.testing.assert_allclose(value[0], expected_value, rtol=1e-5, atol=1e-5)
        if output_derivatives:
            np.testing.assert_allclose(d1, cpu[:, 0], rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(d2, -cpu[:, 1], rtol=1e-5, atol=1e-5)
print(
    "Gamma CUDA simulator: derivative signs, values, zero weights and optional outputs passed"
)
