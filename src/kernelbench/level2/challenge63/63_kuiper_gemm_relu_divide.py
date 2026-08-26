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
_bridge = os.path.join(_dir, 'kuiper_grd_bridge.cu')

kuiper_grd = load(
    name="kuiper_grd_63",
    sources=[_bridge],
    extra_include_paths=[_include_dir, _kbench_include_dir, _obj_dir],
    extra_cuda_cflags=['-DKUIPER_CFG_TENSORCORES=0'],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Verified-Kuiper drop-in for KernelBench L2 #63 (Gemm_ReLU_Divide).
    forward(x) = relu(linear(x)) / divisor.

    The nn.Linear submodule is created identically to the reference Model so
    that, under KernelBench's fixed-seed init, the weight/bias draws match.
    """
    def __init__(self, in_features, out_features, divisor):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.divisor = divisor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.linear.weight
        b = self.linear.bias
        if (x.dim() == 2 and x.dtype == torch.float32 and x.is_cuda
                and W.dtype == torch.float32 and b is not None
                and b.dtype == torch.float32):
            return kuiper_grd.kuiper_grd(x, W, b, float(self.divisor))
        # Reference fallback
        x = self.linear(x)
        x = torch.relu(x)
        return x / self.divisor
