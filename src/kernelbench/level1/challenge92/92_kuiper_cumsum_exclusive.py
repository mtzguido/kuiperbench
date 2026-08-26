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
_bridge = os.path.join(_dir, 'kuiper_cumsum_exclusive_bridge.cu')

kuiper_ext = load(
    name="kuiper_cumsum_exclusive_92",
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
        if self.dim == 1 and x.dim() == 2 and x.dtype == torch.float32 and x.is_cuda:
            return kuiper_ext.kuiper_cumsum_exclusive_dim1(x)
        cumsum = torch.cumsum(
            x.narrow(dim=self.dim, start=0, length=x.size(self.dim) - 1),
            dim=self.dim,
        )
        return torch.cat(
            (torch.zeros_like(x.select(self.dim, 0).unsqueeze(self.dim)), cumsum),
            dim=self.dim,
        )
