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
_bridge = os.path.join(_dir, 'kuiper_msr_bridge.cu')

kuiper_msr = load(
    name="kuiper_msr_40",
    sources=[_bridge],
    extra_include_paths=[_include_dir, _kbench_include_dir, _obj_dir],
    extra_cuda_cflags=['-DKUIPER_CFG_TENSORCORES=0'],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Verified-Kuiper drop-in for KernelBench L2 #40 (Matmul_Scaling_ResidualAdd).
    forward(x) = g * scaling_factor + g,  g = x @ W.T + bias.

    The nn.Linear submodule is created identically to the reference Model so
    that, under KernelBench's fixed-seed init, the weight/bias draws match.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.matmul.weight
        b = self.matmul.bias
        if (x.dim() == 2 and x.dtype == torch.float32 and x.is_cuda
                and W.dtype == torch.float32 and b is not None and b.dim() == 1
                and b.dtype == torch.float32):
            return kuiper_msr.kuiper_msr(x, W, b, float(self.scaling_factor))
        # Reference fallback
        x = self.matmul(x)
        original_x = x.clone().detach()
        x = x * self.scaling_factor
        x = x + original_x
        return x
