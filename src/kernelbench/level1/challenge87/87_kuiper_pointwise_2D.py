"""KernelBench L1 #87: pointwise 2D convolution (1x1 kernel, bias=False).

Reuses the verified Kuiper.KB.Conv2DGeneral primitive (the Conv2D.Naive
kernel) instantiated at kh=kw=1, stride=1, pad=0, dilation=1. A 1x1
convolution is mathematically a per-pixel dot product over the input
channels — it is the simplest non-degenerate Conv2D, and it falls out
of the general path for free.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'challenge62'))
import torch
# Disable TF32 in cuDNN so the PyTorch reference also runs full fp32 conv;
# without this the reference's TF32 (10-bit mantissa) can introduce
# extra error vs our naive f32 accumulator. See challenge62/skeptic.txt.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from _kuiper_conv2d_general import kuiper_conv2d_general
import torch.nn as nn


class ModelNew(nn.Module):
    """KB L1 #87: pointwise (1x1) 2D conv, bias optional (defaults False)."""
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        # Mirror the reference module's weight init: nn.Conv2d(C_in, C_out,
        # kernel_size=1, stride=1, padding=0, bias=bias).
        self.conv1d = nn.Conv2d(in_channels, out_channels,
                                kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return kuiper_conv2d_general(
            x, self.conv1d.weight, self.conv1d.bias,
                                     stride=1, padding=0, dilation=1, groups=1)
