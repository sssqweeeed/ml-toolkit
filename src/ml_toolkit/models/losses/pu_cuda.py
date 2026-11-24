"""CUDA uPU objective, imported only when GPU training is requested."""

import logging
import math

from ._cuda import CudaArray, GPUObjective, cuda


def make_upu_gpu_objective(positive_scale: float) -> GPUObjective:
    driver_logger = logging.getLogger("numba.cuda.cudadrv.driver")
    if driver_logger.level == logging.NOTSET:
        driver_logger.setLevel(logging.WARNING)

    # CatBoost compiles this method with cuda.jit. Capture the immutable scale
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
            if logit >= 0:
                exp_logit = math.exp(-logit)
                probability = 1.0 / (1.0 + exp_logit)
                softplus = logit + math.log1p(exp_logit)
            else:
                exp_logit = math.exp(logit)
                probability = exp_logit / (1.0 + exp_logit)
                softplus = math.log1p(exp_logit)
            # GPU CatBoost expects positive curvature, unlike its CPU API. The
            # whole pool is the unlabeled sample; labeled rows add the scaled
            # positive term.
            der2 = probability * (1.0 - probability)
            # Avoid max(a, b): its variadic Numba overload can fail during
            # CUDA compilation even though the simulator accepts it.
            if target[index] == 1:
                der1 = positive_scale - probability
                value = positive_scale * logit - softplus
            else:
                der1 = -probability
                value = -softplus
            if len(value_output):
                cuda.atomic.add(value_output, 0, weight * value)
            if len(der1_output):
                der1_output[index] = weight * der1
            if len(der2_output):
                der2_output[index] = weight * der2

    return GPUUPUObjective()
