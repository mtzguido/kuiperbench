import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from _kuiper_convt1d_general import kuiper_convt1d_general
import torch.nn as nn

class ModelNew(nn.Module):
    """KB L1 #74: ConvTranspose1D dilated (Kuiper-verified)."""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv1d_transpose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding,
            dilation=dilation, bias=bias)

    def forward(self, x):
        ct = self.conv1d_transpose
        return kuiper_convt1d_general(
            x, ct.weight, ct.bias,
            stride=ct.stride[0], padding=ct.padding[0],
            output_padding=ct.output_padding[0],
            dilation=ct.dilation[0], groups=ct.groups)
