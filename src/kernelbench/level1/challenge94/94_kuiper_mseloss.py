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
_bridge = os.path.join(_dir, 'kuiper_mseloss_bridge.cu')

kuiper_mseloss = load(
    name="kuiper_mseloss_94",
    sources=[_bridge],
    extra_include_paths=[_include_dir, _kbench_include_dir, _obj_dir],
    extra_cuda_cflags=['-DKUIPER_CFG_TENSORCORES=0'],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if (predictions.is_cuda and targets.is_cuda
                and predictions.dtype == torch.float32
                and targets.dtype == torch.float32
                and predictions.shape == targets.shape):
            p_flat = predictions.contiguous().view(-1)
            t_flat = targets.contiguous().view(-1)
            return kuiper_mseloss.kuiper_mseloss(p_flat, t_flat)
        return torch.mean((predictions - targets) ** 2)
