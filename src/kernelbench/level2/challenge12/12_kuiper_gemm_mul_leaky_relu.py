import torch
import torch.nn as nn
import os
from torch.utils.cpp_extension import load

_dir = os.path.dirname(os.path.abspath(__file__))
_kuiper_root = os.path.abspath(os.path.join(_dir, '..', '..', '..', '..'))
_obj_dir = os.path.join(_kuiper_root, 'obj')
_kuiper_home = os.environ.get('KUIPER_HOME', os.path.join(_kuiper_root, '.kuiper'))
_include_dir = os.path.join(_kuiper_home, 'include')
_kbench_include_dir = os.path.join(_kuiper_root, 'include')
_bridge = os.path.join(_dir, 'kuiper_gmlr_bridge.cu')

kuiper_gmlr = load(
    name="kuiper_gmlr_12",
    sources=[_bridge],
    extra_include_paths=[_include_dir, _kbench_include_dir, _obj_dir],
    extra_cuda_cflags=['-DKUIPER_CFG_TENSORCORES=0'],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Verified-Kuiper drop-in for KernelBench L2 #12 (Gemm_Multiply_LeakyReLU).
    forward(x) = leaky_relu((x @ W.T + bias) * multiplier, negative_slope).

    The nn.Linear submodule is created identically to the reference Model so
    that, under KernelBench's fixed-seed init, the weight/bias draws match.
    """
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.multiplier = multiplier
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.gemm.weight
        b = self.gemm.bias
        if (x.dim() == 2 and x.dtype == torch.float32 and x.is_cuda
                and W.dtype == torch.float32 and b is not None and b.dim() == 1
                and b.dtype == torch.float32):
            return kuiper_gmlr.kuiper_gmlr(
                x, W, b, float(self.multiplier), float(self.negative_slope))
        # Reference fallback
        x = self.gemm(x)
        x = x * self.multiplier
        return self.leaky_relu(x)
