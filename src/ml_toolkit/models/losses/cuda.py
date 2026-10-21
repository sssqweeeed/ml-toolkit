"""Shared CUDA interfaces, imported only by GPU implementations."""

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
