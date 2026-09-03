import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
# Disable TF32 in cuDNN so the PyTorch reference also runs full fp32 conv.
# Same reason as L1 #67 and the Conv2D family: without this, cuDNN's TF32
# (10-bit mantissa) introduces ~5e-4 error vs our naive f32 accumulator
# past the 1e-4 fp32 tolerance from get_tolerance_for_precision.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from _kuiper_conv1d_general import kuiper_conv1d_general
import torch.nn as nn

class ModelNew(nn.Module):
    """KB L1 #76: 1D conv with configurable stride and dilation
    (no padding parameter exposed by reference; reference defaults pad=0,
    groups=1, bias=False)."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size,
                                stride=stride, dilation=dilation, bias=bias)
        self._stride = stride
        self._dilation = dilation

    def forward(self, x):
        return kuiper_conv1d_general(x, self.conv1d.weight, self.conv1d.bias,
                                     stride=self._stride, padding=0,
                                     dilation=self._dilation, groups=1)
