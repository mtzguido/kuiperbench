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
_bridge = os.path.join(_dir, 'kuiper_mdg_bridge.cu')

kuiper_mdg = load(
    name="kuiper_mdg_86",
    sources=[_bridge],
    extra_include_paths=[_include_dir, _kbench_include_dir, _obj_dir],
    extra_cuda_cflags=['-DKUIPER_CFG_TENSORCORES=0'],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Verified-Kuiper drop-in for KernelBench L2 #86 (Matmul_Divide_GELU).
    forward(x) = gelu((x @ W.T + bias) / divisor)   (exact erf-form GELU).

    The nn.Linear submodule is created identically to the reference Model so
    that, under KernelBench's fixed-seed init, the weight/bias draws match.
    """
    def __init__(self, input_size, output_size, divisor):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.linear.weight
        b = self.linear.bias
        if (x.dim() == 2 and x.dtype == torch.float32 and x.is_cuda
                and W.dtype == torch.float32 and b is not None and b.dim() == 1
                and b.dtype == torch.float32):
            return kuiper_mdg.kuiper_mdg(x, W, b, float(self.divisor))
        # Reference fallback
        x = self.linear(x)
        x = x / self.divisor
        return torch.nn.functional.gelu(x)
