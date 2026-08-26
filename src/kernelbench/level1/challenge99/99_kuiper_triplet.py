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
_bridge = os.path.join(_dir, 'kuiper_triplet_bridge.cu')

kuiper_triplet = load(
    name="kuiper_triplet_99",
    sources=[_bridge],
    extra_include_paths=[_include_dir, _kbench_include_dir, _obj_dir],
    extra_cuda_cflags=['-DKUIPER_CFG_TENSORCORES=0'],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = float(margin)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        # KernelBench reference: torch.nn.TripletMarginLoss(margin=margin)(a, p, n)
        # which computes mean_b max(0, ||a_b - p_b||_2 - ||a_b - n_b||_2 + margin).
        if (anchor.is_cuda and positive.is_cuda and negative.is_cuda
                and anchor.dtype == torch.float32
                and positive.dtype == torch.float32
                and negative.dtype == torch.float32
                and anchor.dim() == 2 and positive.dim() == 2 and negative.dim() == 2
                and anchor.shape == positive.shape == negative.shape):
            B = anchor.shape[0]
            return kuiper_triplet.kuiper_triplet(
                anchor.contiguous(),
                positive.contiguous(),
                negative.contiguous(),
                self.margin)
        # Fallback for shapes outside the verified path.
        return torch.nn.functional.triplet_margin_loss(anchor, positive, negative, margin=self.margin)
