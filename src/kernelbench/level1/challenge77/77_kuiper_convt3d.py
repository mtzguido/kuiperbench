import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from _kuiper_convt3d_general import kuiper_convt3d_general
import torch.nn as nn


class ModelNew(nn.Module):
    """KB L1 #77: ConvTranspose3D, square + padded/dilated/strided.
    Mirrors the upstream Model.__init__ defaults exactly (notably bias=False)."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 bias: bool = False):
        super().__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels, out_channels,
            kernel_size=(kernel_size, kernel_size, kernel_size),
            stride=stride, padding=padding, dilation=dilation, bias=bias)

    def forward(self, x):
        ct = self.conv_transpose3d
        return kuiper_convt3d_general(
            x, ct.weight, ct.bias,
            stride=ct.stride, padding=ct.padding,
            output_padding=ct.output_padding,
            dilation=ct.dilation, groups=ct.groups)
