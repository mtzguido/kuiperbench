import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from _kuiper_convt3d_general import kuiper_convt3d_general
import torch.nn as nn


class ModelNew(nn.Module):
    """KB L1 #58: ConvTranspose3D, asymmetric input + asymmetric kernel.
    Mirrors the upstream Model.__init__ defaults exactly (notably bias=False)."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0),
                 output_padding: tuple = (0, 0, 0), groups: int = 1,
                 bias: bool = False):
        super().__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding,
            output_padding=output_padding, groups=groups, bias=bias)

    def forward(self, x):
        ct = self.conv_transpose3d
        w = ct.weight.contiguous().to(x.device).to(torch.float32)
        b = None
        if ct.bias is not None:
            b = ct.bias.contiguous().to(x.device).to(torch.float32)
        return kuiper_convt3d_general(
            x, w, b,
            stride=ct.stride, padding=ct.padding,
            output_padding=ct.output_padding,
            dilation=ct.dilation, groups=ct.groups)
