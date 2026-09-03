import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from _kuiper_convt2d_grouped import kuiper_convt2d_grouped
import torch.nn as nn


class ModelNew(nn.Module):
    """KB L1 #75: ConvTranspose2D — asymmetric, strided, grouped, padded,
    dilated.  Mirrors the upstream Model.__init__ defaults exactly
    (notably bias=False).  The Kuiper entry computes the full grouped result."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: tuple = (1, 1), padding: tuple = (0, 0),
                 dilation: tuple = (1, 1), groups: int = 1,
                 bias: bool = False):
        super().__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        ct = self.conv_transpose2d
        return kuiper_convt2d_grouped(
            x, ct.weight, ct.bias,
            stride=ct.stride, padding=ct.padding,
            output_padding=ct.output_padding,
            dilation=ct.dilation, groups=ct.groups)
