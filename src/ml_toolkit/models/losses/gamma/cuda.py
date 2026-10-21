"""Single-device CUDA Gamma objective; risk evaluation uses the CPU metric."""

import logging
import math

from ..cuda import CudaArray, GPUObjective, cuda


def make_gamma_gpu_objective() -> GPUObjective:
    driver_logger = logging.getLogger("numba.cuda.cudadrv.driver")
    if driver_logger.level == logging.NOTSET:
        driver_logger.setLevel(logging.WARNING)

    class GPUGammaObjective:
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
            weight = 1.0 if len(weights) == 0 else weights[index]
            if weight == 0.0:
                if len(der1_output):
                    der1_output[index] = 0.0
                if len(der2_output):
                    der2_output[index] = 0.0
                return
            log_mean = approxes[index]
            z = math.log(target[index]) - log_mean
            ratio = math.exp(z)
            if len(value_output):
                cuda.atomic.add(value_output, 0, -weight * (log_mean + ratio))
            if len(der1_output):
                der1_output[index] = weight * math.expm1(z)
            if len(der2_output):
                # GPU expects positive curvature; CPU expects its negative.
                der2_output[index] = weight * ratio

    return GPUGammaObjective()
