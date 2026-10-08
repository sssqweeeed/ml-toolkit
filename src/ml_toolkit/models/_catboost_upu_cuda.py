"""CUDA uPU objective, imported only when GPU training is requested."""

import math
from typing import Protocol, cast

from numba import cuda as _numba_cuda


class CudaArray(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> float: ...
    def __setitem__(self, index: int, value: float) -> None: ...


class _CudaAtomic(Protocol):
    def add(self, array: CudaArray, index: int, value: float) -> float: ...


class _CudaAPI(Protocol):
    @property
    def atomic(self) -> _CudaAtomic: ...

    def grid(self, ndim: int) -> int: ...


# Numba's CUDA intrinsics have compiler-only signatures. Describe the kernel
# interface here while retaining the actual module for CUDA JIT and simulation.
cuda = cast(_CudaAPI, _numba_cuda)


class GPUObjective(Protocol):
    def calc_ders_range_gpu(
        self,
        approxes: CudaArray,
        target: CudaArray,
        weights: CudaArray,
        value_output: CudaArray,
        der1_output: CudaArray,
        der2_output: CudaArray,
    ) -> None: ...


def make_upu_gpu_objective(
    positive_scale: float, unlabeled_scale: float
) -> GPUObjective:
    # CatBoost compiles this method with cuda.jit. Capture immutable scales
    # rather than accessing Python instance state inside the CUDA kernel.
    class GPUUPUObjective:
        def calc_ders_range_gpu(
            self,
            approxes: CudaArray,
            target: CudaArray,
            weights: CudaArray,
            value_output: CudaArray,
            der1_output: CudaArray,
            der2_output: CudaArray,
        ) -> None:
            index = cuda.grid(1)
            if index >= len(approxes):
                return
            logit = approxes[index]
            weight = 1.0 if len(weights) == 0 else weights[index]
            if target[index] == 1:
                der1 = positive_scale
                der2 = 0.0
                value = positive_scale * logit
            else:
                if logit >= 0:
                    probability = 1.0 / (1.0 + math.exp(-logit))
                else:
                    exp_logit = math.exp(logit)
                    probability = exp_logit / (1.0 + exp_logit)
                der1 = -unlabeled_scale * probability
                # GPU CatBoost expects positive curvature, unlike its CPU API.
                der2 = unlabeled_scale * probability * (1.0 - probability)
                value = -unlabeled_scale * (
                    max(logit, 0.0) + math.log1p(math.exp(-abs(logit)))
                )
            if len(value_output):
                cuda.atomic.add(value_output, 0, weight * value)
            if len(der1_output):
                der1_output[index] = weight * der1
            if len(der2_output):
                der2_output[index] = weight * der2

    return GPUUPUObjective()
