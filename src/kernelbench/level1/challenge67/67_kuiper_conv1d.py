import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
# Disable TF32 in cuDNN so the PyTorch reference also runs full fp32 conv.
# Same reason as L1 #50/#55/#56/#62/#63: without this, cuDNN's TF32 (10-bit
# mantissa) introduces ~5e-4 error vs our naive f32 accumulator past the
# 1e-4 fp32 tolerance from get_tolerance_for_precision.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from _kuiper_conv1d_general import kuiper_conv1d_general
import torch.nn as nn

class ModelNew(nn.Module):
    """KB L1 #67: standard 1D conv (groups=1, configurable params, default
    stride=1, padding=0, dilation=1, bias=False)."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 groups: int = 1, bias: bool = False):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=padding,
                                dilation=dilation, groups=groups, bias=bias)
        self._stride = stride
        self._padding = padding
        self._dilation = dilation
        self._groups = groups

    def forward(self, x):
        return kuiper_conv1d_general(x, self.conv1d.weight, self.conv1d.bias,
                                     stride=self._stride, padding=self._padding,
                                     dilation=self._dilation, groups=self._groups)
