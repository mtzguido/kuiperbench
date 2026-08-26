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
_bridge = os.path.join(_dir, 'kuiper_gar_bridge.cu')

kuiper_gar = load(
    name="kuiper_gar_76",
    sources=[_bridge],
    extra_include_paths=[_include_dir, _kbench_include_dir, _obj_dir],
    extra_cuda_cflags=['-DKUIPER_CFG_TENSORCORES=0'],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Verified-Kuiper drop-in for KernelBench L2 #76 (Gemm_Add_ReLU).
    forward(x) = relu(x @ W.T + bias).

    The nn.Linear(bias=False) submodule and the separate bias Parameter are
    created identically to the reference Model so that, under KernelBench's
    fixed-seed init, the weight/bias draws match.
    """
    def __init__(self, in_features, out_features, bias_shape):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.gemm.weight
        b = self.bias
        if (x.dim() == 2 and x.dtype == torch.float32 and x.is_cuda
                and W.dtype == torch.float32 and b.dim() == 1
                and b.dtype == torch.float32):
            return kuiper_gar.kuiper_gar(x, W, b)
        # Reference fallback
        x = self.gemm(x)
        x = x + self.bias
        return torch.relu(x)
