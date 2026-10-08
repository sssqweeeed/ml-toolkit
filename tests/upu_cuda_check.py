"""Run uPU's CUDA kernel in an isolated Numba CUDA simulator process."""

import inspect

import numpy as np
from numba import cuda

from ml_toolkit.models._catboost_upu import upu_derivatives
from ml_toolkit.models._catboost_upu_cuda import make_upu_gpu_objective

positive_scale, unlabeled_scale = 0.71, 1.37
objective = make_upu_gpu_objective(positive_scale, unlabeled_scale)
method = type(objective).calc_ders_range_gpu
# CatBoost's automatic CUDA JIT requires the instance argument to be unused.
assert inspect.getsource(method).count("self") == 1
kernel = cuda.jit(method)
z = np.array([-1000, 1000, -2, 0, 5, 0.8, -0.8, 1000, -1000], dtype=np.float32)
y = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0], dtype=np.float32)
for weighted in (False, True):
    weights = (
        np.array([1, 0.2, 0, 3, 1, 2, 0.7, 0.5, 1], dtype=np.float32)
        if weighted
        else np.empty(0, dtype=np.float32)
    )
    value = np.zeros(1, dtype=np.float32)
    der1, der2 = np.zeros_like(z), np.zeros_like(z)
    kernel[3, 4](0, z, y, weights, value, der1, der2)
    cpu = upu_derivatives(
        z.astype(float),
        y,
        weights if weighted else None,
        positive_scale,
        unlabeled_scale,
    )
    np.testing.assert_allclose(der1, cpu[:, 0], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(der2, -cpu[:, 1], rtol=1e-6, atol=1e-7)
    terms = np.where(y == 1, positive_scale * z, -unlabeled_scale * np.logaddexp(0, z))
    expected = np.sum(terms * (weights if weighted else 1))
    np.testing.assert_allclose(value[0], expected, rtol=1e-6, atol=1e-4)
    # CatBoost can request a value without derivatives, or derivatives only.
    empty = np.empty(0, dtype=np.float32)
    value[:] = 0
    kernel[3, 4](0, z, y, weights, value, empty, empty)
    np.testing.assert_allclose(value[0], expected, rtol=1e-6, atol=1e-4)
    kernel[3, 4](0, z, y, weights, empty, der1, der2)
    np.testing.assert_allclose(der1, cpu[:, 0], rtol=1e-6, atol=1e-7)
print(
    "CUDA simulator: derivatives, curvature signs, values and optional outputs passed"
)
