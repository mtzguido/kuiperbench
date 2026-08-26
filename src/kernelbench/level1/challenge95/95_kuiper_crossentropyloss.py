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
_bridge = os.path.join(_dir, 'kuiper_crossentropyloss_bridge.cu')

kuiper_crossentropy = load(
    name="kuiper_crossentropy_95",
    sources=[_bridge],
    extra_include_paths=[_include_dir, _kbench_include_dir, _obj_dir],
    extra_cuda_cflags=['-DKUIPER_CFG_TENSORCORES=0'],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # KernelBench reference: torch.nn.functional.cross_entropy(predictions, targets)
        # which computes mean_b ( -log_softmax(predictions[b])[targets[b]] ).
        if (predictions.is_cuda and targets.is_cuda
                and predictions.dtype == torch.float32
                and predictions.dim() == 2
                and targets.dim() == 1
                and predictions.size(0) == targets.size(0)):
            return kuiper_crossentropy.kuiper_crossentropy(
                predictions.contiguous(), targets.contiguous())
        # Fallback for shapes outside the verified path.
        return torch.nn.functional.cross_entropy(predictions, targets)
