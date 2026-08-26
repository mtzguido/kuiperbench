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
_bridge = os.path.join(_dir, 'kuiper_gdss_bridge.cu')

kuiper_gdss = load(
    name="kuiper_gdss_14",
    sources=[_bridge],
    extra_include_paths=[_include_dir, _kbench_include_dir, _obj_dir],
    extra_cuda_cflags=['-DKUIPER_CFG_TENSORCORES=0'],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Verified-Kuiper drop-in for KernelBench L2 #14 (Gemm_Divide_Sum_Scaling).
    forward(x) = sum(matmul(x, W.T) / 2, dim=1, keepdim=True) * scaling_factor.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (x.dim() == 2 and x.dtype == torch.float32 and x.is_cuda
                and self.weight.dtype == torch.float32):
            return kuiper_gdss.kuiper_gdss(x, self.weight, float(self.scaling_factor))
        # Reference fallback
        y = torch.matmul(x, self.weight.T)
        y = y / 2
        y = torch.sum(y, dim=1, keepdim=True)
        return y * self.scaling_factor
