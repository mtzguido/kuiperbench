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
_bridge = os.path.join(_dir, 'kuiper_sdpa_bridge.cu')

kuiper_sdpa = load(
    name="kuiper_sdpa_97",
    sources=[_bridge],
    extra_include_paths=[_include_dir, _kbench_include_dir, _obj_dir],
    extra_cuda_cflags=['-DKUIPER_CFG_TENSORCORES=0'],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        if (Q.is_cuda and K.is_cuda and V.is_cuda
                and Q.dtype == torch.float32
                and K.dtype == torch.float32
                and V.dtype == torch.float32
                and Q.dim() == 4 and Q.shape == K.shape == V.shape):
            return kuiper_sdpa.kuiper_sdpa(Q, K, V)
        return torch.nn.functional.scaled_dot_product_attention(Q, K, V)
