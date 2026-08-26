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
_bridge = os.path.join(_dir, 'kuiper_cumsum_bridge.cu')

kuiper_cumsum = load(
    name="kuiper_cumsum_89",
    sources=[_bridge],
    extra_include_paths=[_include_dir, _kbench_include_dir, _obj_dir],
    extra_cuda_cflags=['-DKUIPER_CFG_TENSORCORES=0'],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The verified kernel implements cumsum(x, dim=1) for a 2-D
        # float32 CUDA input.  Defer to PyTorch otherwise.
        if self.dim == 1 and x.dim() == 2 and x.dtype == torch.float32 and x.is_cuda:
            return kuiper_cumsum.kuiper_cumsum_dim1(x)
        return torch.cumsum(x, dim=self.dim)
