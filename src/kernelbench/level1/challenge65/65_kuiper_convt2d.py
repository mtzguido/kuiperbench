import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from _kuiper_convt2d_general import kuiper_convt2d_general
import torch.nn as nn

class ModelNew(nn.Module):
    """KB L1 #65: ConvTranspose2D (Kuiper-verified, general parameters).

    Defaults match the KernelBench Model class: stride=1, padding=0,
    output_padding=0, dilation=(1,1), groups=1, bias=False.  The KB
    `get_init_inputs()` only passes [in_channels, out_channels, kernel_size]
    so the rest must default consistently with the reference.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0,
                 dilation=1, groups=1, bias=False):
        super().__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding,
            dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        ct = self.conv_transpose2d
        return kuiper_convt2d_general(
            x, ct.weight, ct.bias,
            stride=ct.stride, padding=ct.padding,
            output_padding=ct.output_padding,
            dilation=ct.dilation, groups=ct.groups)
